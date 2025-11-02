import typer
import json
from pathlib import Path
from rich import print

from config import settings
from workflows import get_llm, load_and_split_documents, run_single_round_analysis, run_multi_round_analysis_with_memory, run_multi_round_analysis_without_memory
from langchain.prompts import ChatPromptTemplate

# 创建Typer应用实例
app = typer.Typer(help="模块化、可扩展的LLM文档分析工作流")

def save_results(output_path: Path, filename: str, results: dict):
    """保存结果到txt和json文件（兼容旧格式）"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON结果
    json_filepath = output_path / f"{filename}.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 结构化结果已保存到: [bold green]{json_filepath}[/bold green]")
    
    # 保存可读的TXT报告
    txt_filepath = output_path / f"{filename}.txt"
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(f"文件分析报告: {filename}\n")
        if "memory_type" in results:
            f.write(f"记忆类型: {results['memory_type']}\n")
        f.write("="*50 + "\n\n")
        
        if "single_round_result" in results:
            f.write(results["single_round_result"])
        elif "multi_round_results" in results:
            for round_data in results["multi_round_results"]:
                f.write(f"--- 第 {round_data['round']} 轮分析 ---\n")
                if "memory_type" in round_data:
                    f.write(f"记忆类型: {round_data['memory_type']}\n")
                f.write(f"Prompt模板:\n{round_data['prompt_template']}\n\n")
                f.write(f"分析结果:\n{round_data['result']}\n\n")
                
                # 如果有对话历史，显示对话历史信息
                if "conversation_history" in round_data:
                    if isinstance(round_data["conversation_history"], list):
                        f.write(f"对话历史长度: {len(round_data['conversation_history'])} 条消息\n")
                    else:
                        f.write(f"对话历史: {round_data['conversation_history']}\n")
                
                f.write("="*50 + "\n\n")
    print(f"✅ 详细文本报告已保存到: [bold green]{txt_filepath}[/bold green]")


def save_file_results(output_path: Path, file_result: dict):
    """保存单个文件的分析结果"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    source_file = file_result["source_file"]
    file_path = Path(source_file)
    filename = file_path.stem  # 获取不带扩展名的文件名
    
    # 保存JSON结果
    json_filepath = output_path / f"{filename}.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(file_result, f, ensure_ascii=False, indent=2)
    print(f"✅ 文件 {filename} 的JSON结果已保存到: [bold green]{json_filepath}[/bold green]")
    
    # 保存TXT报告
    txt_filepath = output_path / f"{filename}.txt"
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(f"文件分析报告: {filename}\n")
        f.write(f"源文件: {source_file}\n")
        if "memory_type" in file_result:
            f.write(f"记忆类型: {file_result['memory_type']}\n")
        f.write("="*50 + "\n\n")
        
        for round_data in file_result["analysis_history"]:
            round_type = round_data.get('round_type', f'第{round_data["round"]}轮')
            f.write(f"--- 第 {round_data['round']} 轮分析 ({round_type}) ---\n")
            f.write(f"分析结果:\n{round_data['result']}\n\n")
            
            # 显示对话历史信息
            if "conversation_history" in round_data:
                if isinstance(round_data["conversation_history"], list):
                    f.write(f"对话历史长度: {len(round_data['conversation_history'])} 条消息\n")
                else:
                    f.write(f"对话历史: {round_data['conversation_history']}\n")
            
            f.write("="*50 + "\n\n")
    print(f"✅ 文件 {filename} 的文本报告已保存到: [bold green]{txt_filepath}[/bold green]")


@app.command(name="single-round", help="对指定目录的文档进行单轮批量分析。")
def single_round():
    """执行单轮分析"""
    print("🚀 [bold blue]启动单轮分析工作流...[/bold blue]")
    
    # 1. 加载配置和初始化LLM
    llm = get_llm(settings.llm_provider)
    output_path = settings.paths.output_dir / "single_round"
    
    # 2. 加载并分割文档
    print(f"📂 正在从 [yellow]{settings.paths.input_dir}[/yellow] 加载文档...")
    docs = load_and_split_documents(settings)
    print(f"📄 共加载并分割成 {len(docs)} 个文档块。")

    # 3. 加载Prompt模板
    try:
        # 读取prompt1.txt文件
        prompt1_path = settings.paths.prompt_dir / "prompt1.txt"
        prompt_text = prompt1_path.read_text(encoding='utf-8')
        # 注意：这里的模板需要包含 {text} 变量供 map_reduce 链使用
        prompt_template = ChatPromptTemplate.from_template(prompt_text)
    except FileNotFoundError:
        print(f"❌ [bold red]错误: Prompt文件未找到 -> {prompt1_path}[/bold red]")
        raise typer.Exit(code=1)

    # 4. 执行分析
    print("🧠 正在调用LLM进行分析（可能需要一些时间）...")
    file_results = run_single_round_analysis(docs, prompt_template, llm)
    
    # 5. 单独保存每个文件的结果
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file_result in file_results:
        source = file_result["source"]
        result = file_result["result"]
        
        # 从文件路径中提取文件名
        file_path = Path(source)
        filename = file_path.stem  # 获取不带扩展名的文件名
        
        # 保存单个文件的JSON结果
        json_filepath = output_path / f"{filename}.json"
        file_data = {
            "source_file": source,
            "prompt_file": str(settings.paths.prompt_dir / "prompt1.txt"),
            "analysis_result": result
        }
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 文件 {filename} 的JSON结果已保存到: [bold green]{json_filepath}[/bold green]")
        
        # 保存单个文件的TXT报告
        txt_filepath = output_path / f"{filename}.txt"
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"文件分析报告: {filename}\n")
            f.write(f"源文件: {source}\n")
            f.write("="*50 + "\n\n")
            f.write(result)
        print(f"✅ 文件 {filename} 的文本报告已保存到: [bold green]{txt_filepath}[/bold green]")
    
    print(f"\n🎉 [bold green]单轮分析完成！共处理了 {len(file_results)} 个文件。[/bold green]")


@app.command(name="multi-round", help="对文档进行多轮、递进式分析（使用完整对话记忆）。")
def multi_round():
    """执行多轮分析（使用完整对话记忆）"""
    print("🚀 [bold magenta]启动多轮分析工作流（完整对话记忆）...[/bold magenta]")
    
    # 1. 加载配置和初始化LLM
    llm = get_llm(settings.llm_provider)
    output_path = settings.paths.output_dir / "multi_round"
    prompt_files = sorted(settings.paths.prompt_dir.glob("round*.txt"))

    if not prompt_files:
        print(f"❌ [bold red]错误: 在 {settings.paths.prompt_dir} 中未找到任何 'round*.txt' 文件。[/bold red]")
        raise typer.Exit(code=1)

    # 2. 加载并分割文档
    print(f"📂 正在从 [yellow]{settings.paths.input_dir}[/yellow] 加载文档...")
    docs = load_and_split_documents(settings)
    print(f"📄 共加载并分割成 {len(docs)} 个文档块。")

    # 3. 加载所有轮次的Prompt模板
    # 注意：使用对话记忆时，模板不需要包含 {previous_results} 变量
    round_prompts = []
    print("📝 加载Prompt模板:")
    for pf in prompt_files:
        print(f"  - {pf.name}")
        round_prompts.append(ChatPromptTemplate.from_template(pf.read_text(encoding='utf-8')))
        
    # 4. 执行多轮分析（使用完整对话记忆）
    print("🧠 使用完整对话记忆进行多轮分析...")
    final_results_list = run_multi_round_analysis_with_memory(
        docs, round_prompts, llm, memory_type="buffer"
    )
    
    # 5. 结果已实时保存，这里只显示完成信息
    print(f"\n🎉 [bold green]多轮分析完成！共处理了 {len(final_results_list)} 个文件。（使用完整对话记忆）[/bold green]")
    print(f"📁 所有结果已保存到: [bold green]{output_path}[/bold green]")


@app.command(name="multi-round-advanced", help="对文档进行多轮分析，支持选择不同的对话记忆类型。")
def multi_round_advanced(
    memory_type: str = typer.Option(
        "buffer", 
        "--memory-type", 
        "-m", 
        help="对话记忆类型: buffer(完整记忆), window(窗口记忆), summary(摘要记忆)"
    )
):
    """执行多轮分析（支持选择记忆类型）"""
    print(f"🚀 [bold magenta]启动多轮分析工作流（记忆类型: {memory_type}）...[/bold magenta]")
    
    # 1. 加载配置和初始化LLM
    llm = get_llm(settings.llm_provider)
    output_path = settings.paths.output_dir / f"multi_round_{memory_type}"
    prompt_files = sorted(settings.paths.prompt_dir.glob("round*.txt"))

    if not prompt_files:
        print(f"❌ [bold red]错误: 在 {settings.paths.prompt_dir} 中未找到任何 'round*.txt' 文件。[/bold red]")
        raise typer.Exit(code=1)

    # 2. 加载并分割文档
    print(f"📂 正在从 [yellow]{settings.paths.input_dir}[/yellow] 加载文档...")
    docs = load_and_split_documents(settings)
    print(f"📄 共加载并分割成 {len(docs)} 个文档块。")

    # 3. 加载所有轮次的Prompt模板
    round_prompts = []
    print("📝 加载Prompt模板:")
    for pf in prompt_files:
        print(f"  - {pf.name}")
        round_prompts.append(ChatPromptTemplate.from_template(pf.read_text(encoding='utf-8')))
        
    # 4. 执行多轮分析（使用指定记忆类型）
    print(f"🧠 使用 {memory_type} 记忆类型进行多轮分析...")
    final_results_list = run_multi_round_analysis_with_memory(
        docs, round_prompts, llm, memory_type=memory_type
    )
    
    # 5. 结果已实时保存，这里只显示完成信息
    print(f"\n🎉 [bold green]多轮分析完成！共处理了 {len(final_results_list)} 个文件。（使用 {memory_type} 记忆类型）[/bold green]")
    print(f"📁 所有结果已保存到: [bold green]{output_path}[/bold green]")


@app.command(name="multi-round-no-memory", help="对文档进行多轮分析，不使用对话记忆，通过显式传递之前轮次结果。")
def multi_round_no_memory(
    max_concurrent: int = typer.Option(
        5, 
        "--max-concurrent", 
        "-c", 
        help="最大并发数量，控制同时处理的文件数量（默认: 5）"
    )
):
    """执行多轮分析（不使用对话记忆，显式传递之前轮次结果）"""
    print("🚀 [bold cyan]启动多轮分析工作流（无记忆模式）...[/bold cyan]")
    
    # 1. 加载配置和初始化LLM
    llm = get_llm(settings.llm_provider)
    output_path = settings.paths.output_dir / "multi_round_no_memory"
    prompt_files = sorted(settings.paths.prompt_dir.glob("round*.txt"))

    if not prompt_files:
        print(f"❌ [bold red]错误: 在 {settings.paths.prompt_dir} 中未找到任何 'round*.txt' 文件。[/bold red]")
        raise typer.Exit(code=1)

    # 2. 加载并分割文档
    print(f"📂 正在从 [yellow]{settings.paths.input_dir}[/yellow] 加载文档...")
    docs = load_and_split_documents(settings)
    print(f"📄 共加载并分割成 {len(docs)} 个文档块。")

    # 3. 加载所有轮次的Prompt模板
    # 注意：无记忆模式下，模板可以包含 {previous_results} 变量来引用之前轮次的结果
    round_prompts = []
    print("📝 加载Prompt模板:")
    for pf in prompt_files:
        print(f"  - {pf.name}")
        round_prompts.append(ChatPromptTemplate.from_template(pf.read_text(encoding='utf-8')))
        
    # 4. 执行多轮分析（不使用对话记忆）
    print(f"🧠 使用无记忆模式进行多轮分析（显式传递之前轮次结果，最大并发: {max_concurrent}）...")
    final_results_list = run_multi_round_analysis_without_memory(
        docs, round_prompts, llm, max_concurrent
    )
    
    # 5. 结果已实时保存，这里只显示完成信息
    print(f"\n🎉 [bold green]多轮分析完成！共处理了 {len(final_results_list)} 个文件。（无记忆模式）[/bold green]")
    print(f"📁 所有结果已保存到: [bold green]{output_path}[/bold green]")


if __name__ == "__main__":
    app()