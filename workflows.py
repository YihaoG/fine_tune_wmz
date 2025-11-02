# workflows.py

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any
import asyncio

from config import AppSettings, LlmProviderSettings

def get_llm(llm_settings: LlmProviderSettings) -> ChatOpenAI:
    """根据配置初始化并返回一个ChatOpenAI实例"""
    return ChatOpenAI(
        model=llm_settings.model_name,
        temperature=llm_settings.temperature,
        api_key=llm_settings.api_key,
        base_url=llm_settings.base_url
    )

def load_and_split_documents(settings: AppSettings) -> List[Document]:
    """加载并分割文档"""
    # 创建两个加载器：一个用于txt文件，一个用于md文件
    txt_loader = DirectoryLoader(
        settings.paths.input_dir,
        glob="*.txt",  # 读取txt文件
        show_progress=True,
        use_multithreading=True
    )
    
    md_loader = DirectoryLoader(
        settings.paths.input_dir,
        glob="*.md",  # 读取md文件
        show_progress=True,
        use_multithreading=True
    )
    
    # 加载所有文档
    txt_documents = txt_loader.load()
    md_documents = md_loader.load()
    documents = txt_documents + md_documents
    
    # 验证加载的文档内容
    for doc in documents:
        print(f"📄 加载文档: {doc.metadata.get('source', 'unknown')}")
        print(f"   内容长度: {len(doc.page_content)} 字符")
        print(f"   内容预览: {doc.page_content[:200]}...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.processing.chunk_size,
        chunk_overlap=settings.processing.chunk_overlap
    )
    return text_splitter.split_documents(documents)


def run_single_round_analysis(docs: List[Document], prompt_template: ChatPromptTemplate, llm: ChatOpenAI) -> List[Dict[str, str]]:
    """
    运行单轮分析工作流。
    对每个文档分别进行分析，返回每个文件的单独结果。
    """
    file_results = []
    
    # 按文档源文件分组
    docs_by_source = {}
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    print(f"📁 发现 {len(docs_by_source)} 个源文件")
    
    # 对每个源文件分别处理
    for source, source_docs in docs_by_source.items():
        print(f"📄 正在处理文件: {source}")
        
        # 合并同一文件的所有文档块
        combined_text = "\n\n".join([doc.page_content for doc in source_docs])
        print(f"   合并后文本长度: {len(combined_text)} 字符")
        print(f"   文本预览: {combined_text[:300]}...")
        
        # 创建包含合并文本的完整prompt
        full_prompt = prompt_template.format(text=combined_text)
        print(f"   完整prompt长度: {len(str(full_prompt))} 字符")
        
        # 直接调用LLM
        print(f"   正在调用LLM...")
        response = llm.invoke(full_prompt)
        file_result = response.content
        print(f"   LLM返回结果长度: {len(file_result)} 字符")
        print(f"   结果预览: {file_result[:300]}...")
        
        # 保存单个文件的结果
        file_results.append({
            "source": source,
            "result": file_result
        })
    
    return file_results


def run_multi_round_analysis(
    docs: List[Document],
    round_prompts: List[ChatPromptTemplate],
    llm: ChatOpenAI
) -> List[Dict[str, Any]]:
    """
    运行多轮、递进式分析工作流。
    每一轮都以上一轮的结果为基础，保持完整的对话记忆。
    """
    analysis_history = []
    
    # 创建对话记忆
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    for i, prompt_template in enumerate(round_prompts):
        print(f"\n===== 开始第 {i+1} 轮分析... =====")
        
        if i == 0:
            # 第一轮：使用标准的map_reduce链处理文档
            round_results = run_single_round_analysis(docs, prompt_template, llm)
            # 将多个文件的结果合并为一个字符串
            round_result_text = "\n\n".join([result["result"] for result in round_results])
            
            # 获取prompt模板的原始文本内容
            prompt_text = ""
            for message in prompt_template.messages:
                if hasattr(message, 'prompt'):
                    prompt_text += message.prompt.template
                elif hasattr(message, 'template'):
                    prompt_text += message.template
            
            # 使用用户提供的prompt模板，将文档内容替换{text}变量
            current_prompt = prompt_text.replace('{text}', round_result_text).replace('{previous_results}', '')
            print(f"   第一轮Prompt预览: {current_prompt[:300]}...")
            response = conversation.predict(input=current_prompt)
            round_result_text = response
            
        else:
            # 第二轮和第三轮：使用对话链，保持对话记忆
            # 获取prompt模板的原始文本内容
            prompt_text = ""
            for message in prompt_template.messages:
                if hasattr(message, 'prompt'):
                    prompt_text += message.prompt.template
                elif hasattr(message, 'template'):
                    prompt_text += message.template
            
            # 创建当前轮次的提示
            current_prompt = prompt_text.replace('{previous_results}', '').replace('{text}', '')
            
            # 使用对话链，LLM会自动记住之前的对话历史
            response = conversation.predict(input=current_prompt)
            round_result_text = response
        
        # 记录本轮结果
        round_info = {
            "round": i + 1,
            "prompt_template": str(prompt_template.messages),
            "result": round_result_text,
            "conversation_history": memory.chat_memory.messages  # 保存对话历史
        }
        analysis_history.append(round_info)
        
        print(f"   对话历史长度: {len(memory.chat_memory.messages)} 条消息")
        print(f"   当前轮次结果长度: {len(round_result_text)} 字符")
        print(f"   结果预览: {round_result_text[:300]}...")
    
    return analysis_history


def run_multi_round_analysis_without_memory(
    docs: List[Document],
    round_prompts: List[ChatPromptTemplate],
    llm: ChatOpenAI,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    运行多轮、递进式分析工作流，不使用对话记忆。
    通过显式传递之前轮次的结果来实现多轮分析。
    对每个文件分别进行处理，每个文件生成独立的结果。
    
    Args:
        docs: 文档列表
        round_prompts: 每轮的提示模板
        llm: LLM实例
    """
    # 1) 按文档源文件分组，并准备每个文件的合并文本与状态
    docs_by_source: Dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    print(f"📁 发现 {len(docs_by_source)} 个源文件，将每轮并行处理各文件（最大并发数: {max_concurrent}）")

    file_states: Dict[str, Dict[str, Any]] = {}
    for source, source_docs in docs_by_source.items():
        combined_text = "\n\n".join([doc.page_content for doc in source_docs])
        print(f"\n📄 准备文件: {source}")
        print(f"   合并后文本长度: {len(combined_text)} 字符")
        file_states[source] = {
            "combined_text": combined_text,
            "previous_results": [],
            "analysis_history": []
        }



    # 4) 按批次处理：每个批次处理完所有轮次后再处理下一批次
    sources = list(file_states.keys())
    num_rounds = len(round_prompts)
    
    # 分批处理，控制并发数量
    for batch_start in range(0, len(sources), max_concurrent):
        batch_sources = sources[batch_start:batch_start + max_concurrent]
        batch_num = batch_start // max_concurrent + 1
        print(f"\n===== 开始处理批次 {batch_num}: {len(batch_sources)} 个文件 =====")
        
        # 为当前批次的所有文件处理所有轮次
        async def process_batch_all_rounds() -> Dict[str, List[Dict[str, Any]]]:
            batch_results = {}
            
            # 为每个文件创建任务：处理所有轮次
            async def process_file_all_rounds(source: str) -> List[Dict[str, Any]]:
                file_analysis_history = []
                file_previous_results = []
                
                for round_index in range(num_rounds):
                    prompt_template = round_prompts[round_index]
                    
                    # 构建当前轮次的 prompt
                    prompt_text = ""
                    for message in prompt_template.messages:
                        if hasattr(message, 'prompt'):
                            prompt_text += message.prompt.template
                        elif hasattr(message, 'template'):
                            prompt_text += message.template
                    
                    if round_index == 0:
                        current_prompt = prompt_text.replace('{text}', file_states[source]["combined_text"]).replace('{previous_results}', '')
                        print(f"   📄 {source} 第{round_index+1}轮Prompt预览: {current_prompt[:100]}...")
                    else:
                        current_prompt = prompt_text.replace('{text}', file_states[source]["combined_text"])
                        
                        # 替换特定轮次变量
                        for j, prev_result in enumerate(file_previous_results):
                            round_var = f"{{round{j+1}_result}}"
                            if round_var in current_prompt:
                                current_prompt = current_prompt.replace(round_var, prev_result)
                        
                        # 替换通用变量
                        if "{previous_results}" in current_prompt:
                            previous_results_text = ""
                            for j, prev_result in enumerate(file_previous_results):
                                round_types = ["基础分析", "深度分析", "综合总结", "最终建议", "补充分析"]
                                round_type = round_types[j] if j < len(round_types) else f"第{j+1}轮"
                                previous_results_text += f"\n{'='*60}\n"
                                previous_results_text += f"📋 第 {j+1} 轮分析结果 ({round_type})\n"
                                previous_results_text += f"{'='*60}\n"
                                previous_results_text += f"{prev_result}\n"
                                previous_results_text += f"\n{'='*60}\n"
                            current_prompt = current_prompt.replace('{previous_results}', previous_results_text)
                        
                        print(f"   📄 {source} 第{round_index+1}轮Prompt预览: {current_prompt[:100]}...")
                    
                    # 调用LLM（含重试）
                    max_retries = 10
                    retry_count = 0
                    round_result_text = ""
                    
                    while retry_count < max_retries:
                        try:
                            response = await llm.ainvoke(current_prompt)
                            round_result_text = response.content
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"   ❌ {source} 第{round_index+1}轮重试上限，异常: {e}")
                                round_result_text = f"[发生异常：{e}]"
                                break
                            print(f"   ⚠️ {source} 第{round_index+1}轮第{retry_count}次重试：请求异常，重试中...")
                            continue
                        
                        if len(round_result_text.strip()) == 0:
                            retry_count += 1
                            print(f"   ⚠️ {source} 第{round_index+1}轮第{retry_count}次重试：结果为空，重试中...")
                            if retry_count >= max_retries:
                                print(f"   ❌ {source} 第{round_index+1}轮经{max_retries}次重试仍为空")
                                round_result_text = f"[结果为空：经过{max_retries}次重试仍无结果]"
                                break
                            continue
                        
                        break
                    
                    # 记录本轮结果
                    round_types = ["基础分析", "深度分析", "综合总结", "最终建议", "补充分析"]
                    round_type = round_types[round_index] if round_index < len(round_types) else f"第{round_index+1}轮"
                    round_info = {
                        "round": round_index + 1,
                        "round_type": round_type,
                        "result": round_result_text
                    }
                    
                    file_analysis_history.append(round_info)
                    file_previous_results.append(round_result_text)
                    
                    print(f"   ✅ {source} 第{round_index+1}轮完成，结果长度: {len(round_result_text)}")
                
                return file_analysis_history
            
            # 并发处理当前批次的所有文件
            batch_tasks = [process_file_all_rounds(source) for source in batch_sources]
            batch_results_list = await asyncio.gather(*batch_tasks)
            
            # 收集结果
            for source, analysis_history in zip(batch_sources, batch_results_list):
                batch_results[source] = analysis_history
            
            return batch_results
        
        # 处理当前批次的所有轮次
        batch_results: Dict[str, List[Dict[str, Any]]] = asyncio.run(process_batch_all_rounds())
        
        # 更新文件状态并立即保存当前批次的结果
        for source, analysis_history in batch_results.items():
            file_states[source]["analysis_history"] = analysis_history
            file_states[source]["previous_results"] = [round_info["result"] for round_info in analysis_history]
            print(f"✅ 批次 {batch_num} 文件 {source} 所有轮次处理完成")
        
        # 立即保存当前批次的所有文件结果
        try:
            from main import save_file_results
            from config import settings
            output_path = settings.paths.output_dir / "multi_round_no_memory"
            
            for source in batch_sources:
                file_result = {
                    "source_file": source,
                    "analysis_history": file_states[source]["analysis_history"]
                }
                save_file_results(output_path, file_result)
                print(f"💾 批次 {batch_num} 文件 {source} 结果已保存")
        except Exception as e:
            print(f"⚠️ 保存批次 {batch_num} 结果时出错: {e}")
        
        print(f"🎉 批次 {batch_num} 完成，共处理 {len(batch_sources)} 个文件的所有 {num_rounds} 轮分析，结果已保存")

    # 5) 组织输出（结果已在每个批次处理完后保存）
    all_file_results: List[Dict[str, Any]] = []
    for source, state in file_states.items():
        file_result = {
            "source_file": source,
            "analysis_history": state["analysis_history"]
        }
        all_file_results.append(file_result)

    print("✅ 所有文件处理完成（无记忆模式，并行按文件，每批次已保存）")
    return all_file_results


def run_multi_round_analysis_with_memory(
    docs: List[Document],
    round_prompts: List[ChatPromptTemplate],
    llm: ChatOpenAI,
    memory_type: str = "buffer",
    max_token_limit: int = 4000
) -> List[Dict[str, Any]]:
    """
    运行多轮、递进式分析工作流，支持不同类型的对话记忆。
    对每个文件分别进行处理，每个文件生成独立的结果。
    
    Args:
        docs: 文档列表
        round_prompts: 每轮的提示模板
        llm: LLM实例
        memory_type: 记忆类型 ("buffer", "window", "summary")
        max_token_limit: 最大token限制
    """
    # 按文档源文件分组
    docs_by_source = {}
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    print(f"📁 发现 {len(docs_by_source)} 个源文件，将分别处理每个文件")
    
    all_file_results = []
    
    # 对每个源文件分别处理
    for source, source_docs in docs_by_source.items():
        print(f"\n📄 正在处理文件: {source}")
        
        # 为每个文件创建独立的对话记忆
        if memory_type == "buffer":
            from langchain.memory import ConversationBufferMemory
            memory = ConversationBufferMemory()
        elif memory_type == "window":
            from langchain.memory import ConversationBufferWindowMemory
            memory = ConversationBufferWindowMemory(k=5)
        elif memory_type == "summary":
            from langchain.memory import ConversationSummaryMemory
            memory = ConversationSummaryMemory(llm=llm)
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
        
        # 为每个文件创建独立的对话链
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        file_analysis_history = []
        
        # 合并同一文件的所有文档块
        combined_text = "\n\n".join([doc.page_content for doc in source_docs])
        print(f"   合并后文本长度: {len(combined_text)} 字符")
        
        for i, prompt_template in enumerate(round_prompts):
            print(f"   ===== 开始第 {i+1} 轮分析... =====")
            
            # 获取prompt模板的原始文本内容
            prompt_text = ""
            for message in prompt_template.messages:
                if hasattr(message, 'prompt'):
                    prompt_text += message.prompt.template
                elif hasattr(message, 'template'):
                    prompt_text += message.template
            
            if i == 0:
                # 第一轮：使用用户提供的prompt模板，将文档内容替换{text}变量
                current_prompt = prompt_text.replace('{text}', combined_text).replace('{previous_results}', '')
                print(f"   第一轮Prompt预览: {current_prompt[:30]}...")
                response = conversation.predict(input=current_prompt)
                round_result_text = response
                
            else:
                # 后续轮次：使用对话记忆，移除{text}和{previous_results}变量
                current_prompt = prompt_text.replace('{previous_results}', '').replace('{text}', '')
                print(f"   第{i+1}轮Prompt预览: {current_prompt[:30]}...")
                response = conversation.predict(input=current_prompt)
                round_result_text = response
            
            # 检查结果长度，如果为空则重试
            max_retries = 3
            retry_count = 0
            original_result = round_result_text
            
            while retry_count < max_retries and len(round_result_text.strip()) == 0:
                retry_count += 1
                print(f"   ⚠️ 第{retry_count}次重试：第{i+1}轮结果为空，重新运行...")
                response = conversation.predict(input=current_prompt)
                round_result_text = response.content
                
                if retry_count >= max_retries and len(round_result_text.strip()) == 0:
                    print(f"   ❌ 第{i+1}轮分析失败：经过{max_retries}次重试仍无结果")
                    round_result_text = f"[第{i+1}轮分析失败：LLM返回空结果]"
            
            # 记录本轮结果（避免JSON序列化问题）
            round_info = {
                "round": i + 1,
                "result": round_result_text,
                "memory_type": memory_type
            }
            
            # 安全地处理对话历史，避免JSON序列化问题
            if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
                # 将Message对象转换为可序列化的字典
                conversation_history = []
                for msg in memory.chat_memory.messages:
                    conversation_history.append({
                        "type": msg.__class__.__name__,
                        "content": msg.content
                    })
                round_info["conversation_history"] = conversation_history
                print(f"   对话历史长度: {len(conversation_history)} 条消息")
            else:
                round_info["conversation_history"] = "摘要记忆" if memory_type == "summary" else []
                print(f"   使用{memory_type}记忆")
            
            file_analysis_history.append(round_info)
            print(f"   当前轮次结果长度: {len(round_result_text)} 字符")
            print(f"   结果预览: {round_result_text[:200]}...")
        
        # 保存单个文件的结果
        file_result = {
            "source_file": source,
            "memory_type": memory_type,
            "analysis_history": file_analysis_history
        }
        all_file_results.append(file_result)
        
        # 立即保存当前文件的结果
        try:
            from main import save_file_results
            from pathlib import Path
            from config import settings
            
            # 创建输出目录
            output_path = settings.paths.output_dir / f"multi_round_{memory_type}"
            save_file_results(output_path, file_result)
            print(f"💾 文件 {source} 结果已立即保存")
        except Exception as e:
            print(f"⚠️ 保存文件 {source} 时出错: {e}")
        
        print(f"✅ 文件 {source} 处理完成")
    
    return all_file_results


def example_usage_with_memory():
    """
    使用示例：展示如何使用不同的对话记忆类型
    """
    from config import AppSettings, LlmProviderSettings
    
    # 配置设置
    settings = AppSettings()
    llm_settings = LlmProviderSettings()
    llm = get_llm(llm_settings)
    
    # 加载文档
    docs = load_and_split_documents(settings)
    
    # 定义多轮提示模板
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # 第一轮：基础分析
    round1_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个专业的文档分析师。请仔细分析提供的文档内容。"),
        HumanMessage(content="请分析以下文档内容：\n\n{text}")
    ])
    
    # 第二轮：深入分析
    round2_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="基于之前的分析，请进行更深入的分析。"),
        HumanMessage(content="请基于之前的分析结果，进一步深入分析文档中的关键问题和趋势。")
    ])
    
    # 第三轮：总结和建议
    round3_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="基于所有之前的分析，请提供总结和建议。"),
        HumanMessage(content="请总结所有分析结果，并提供具体的建议和改进方案。")
    ])
    
    round_prompts = [round1_prompt, round2_prompt, round3_prompt]
    
    print("=== 使用完整对话记忆 ===")
    results_buffer = run_multi_round_analysis_with_memory(
        docs, round_prompts, llm, memory_type="buffer"
    )
    
    print("\n=== 使用窗口对话记忆（保留最近5轮） ===")
    results_window = run_multi_round_analysis_with_memory(
        docs, round_prompts, llm, memory_type="window"
    )
    
    print("\n=== 使用摘要对话记忆 ===")
    results_summary = run_multi_round_analysis_with_memory(
        docs, round_prompts, llm, memory_type="summary"
    )
    
    return {
        "buffer_memory": results_buffer,
        "window_memory": results_window,
        "summary_memory": results_summary
    }