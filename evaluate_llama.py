import torch
import pandas as pd
import nltk
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

class PoemAnalyzer:
    """Class to analyze Vietnamese poems using the Llama3.2 model"""
    def __init__(self):
        # Define the device attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # First, identify which base model this adapter was trained on
        peft_config = PeftConfig.from_pretrained("kienhoang123/Llama3.2_Poem_Analysis")
        base_model_name = peft_config.base_model_name_or_path
        
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load the adapter on top of the model
        self.model = PeftModel.from_pretrained(
            self.model,
            "kienhoang123/Llama3.2_Poem_Analysis",
            is_trainable=False
        )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("kienhoang123/Llama3.2_Poem_Analysis")
    
    def analyze(self, poem, max_new_tokens=100):
        """Analyze a poem to extract key elements"""
        input_text = f"""
        ### Instruction:
        Analyze the given poem and extract its emotional tone, metaphor, setting, motion, and generate a prompt based on the poem.

        ### Input Poem:
        {poem}

        ### Poem Analysis:
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2
        )
        full_analysis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return full_analysis
    
    def extract_elements(self, full_analysis):
        """Extract key elements from the analysis"""
        elements = {
            "emotional_tone": "",
            "metaphor": "",
            "setting": "",
            "motion": ""
        }
        
        lines = full_analysis.split('\n')
        for line in lines:
            line = line.strip()
            if "Emotional Tone:" in line:
                elements["emotional_tone"] = line.split("Emotional Tone:")[1].strip()
            elif "Metaphor:" in line:
                elements["metaphor"] = line.split("Metaphor:")[1].strip()
            elif "Setting:" in line:
                elements["setting"] = line.split("Setting:")[1].strip()
            elif "Motion:" in line:
                elements["motion"] = line.split("Motion:")[1].strip()
        
        # Create a concise analysis in Vietnamese
        concise_analysis = f"""
        Cảm xúc: {elements['emotional_tone']}
        Ẩn dụ: {elements['metaphor']}
        Bối cảnh: {elements['setting']}
        Chuyển động: {elements['motion']}
        """
        
        return concise_analysis


class PromptGenerator:
    """Class to generate image generation prompts from poem analysis"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, analysis):
        """Generate prompt using Ollama API"""
        prompt = f"""
        Bạn là một nghệ sĩ AI chuyên tạo ra các prompt cho mô hình AI tạo hình ảnh (Midjourney, Stable Diffusion).
        Dựa trên phân tích bài thơ sau đây, hãy tạo ra một prompt ngắn gọn bằng tiếng Việt để mô hình AI có thể tạo ra hình ảnh thể hiện được không khí, cảm xúc và ẩn dụ của bài thơ.
        
        Phân tích bài thơ: {analysis}
        
        Prompt cần bao gồm:
        1. Mô tả cảnh vật/khung cảnh chính (dưới 20 từ)
        2. Cảm xúc và không khí (dưới 10 từ)
        3. Màu sắc chủ đạo (dưới 10 từ)
        4. Phong cách nghệ thuật (chỉ 1-2 từ)
        
        LƯU Ý QUAN TRỌNG: 
        - Prompt phải ngắn gọn, dưới 70 từ tổng cộng, không sử dụng từ tiếng Anh.
        - Trả lời NGAY với PROMPT THÔI, không thêm giải thích hay thông tin khác.
        - KHÔNG bắt đầu bằng "Dựa trên phân tích bài thơ..." hoặc "Tôi tạo ra prompt...".
        """
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
        }
        
        try:
            response = requests.post(
                url='http://localhost:11434/api/generate', json=payload
            ).json()
            raw_response = response['response']
            
            # Clean the response to extract just the prompt
            # Remove common prefixes
            prefixes_to_remove = [
                "Dựa trên phân tích bài thơ, tôi tạo ra một prompt sau đây:",
                "Dựa trên phân tích bài thơ, tôi đề xuất prompt sau:",
                "Dựa trên phân tích bài thơ, tôi sẽ tạo ra một prompt ngắn gọn như sau:",
                "Dưới đây là prompt ngắn gọn mà tôi tạo ra dựa trên phân tích bài thơ:",
                "Tôi có thể giúp bạn tạo ra một prompt ngắn gọn để mô hình AI tạo hình ảnh cho bài thơ. Dưới đây là gợi ý của tôi:",
                "Tôi có thể giúp bạn tạo ra một prompt ngắn gọn cho mô hình AI tạo hình ảnh. Dưới đây là gợi ý:",
                "Hy vọng prompt này sẽ giúp mô hình AI tạo ra một hình ảnh phù hợp với cảm xúc và ẩn dụ của bài thơ.",
                "Hy vọng này prompt sẽ giúp mô hình AI tạo ra một hình ảnh thể hiện được không khí và ẩn dụ của bài thơ."
            ]
            
            clean_prompt = raw_response
            for prefix in prefixes_to_remove:
                if prefix in clean_prompt:
                    clean_prompt = clean_prompt.replace(prefix, "").strip()
            
            # Remove quotes if present
            clean_prompt = clean_prompt.strip('"\'')
            
            # Remove any trailing notes or parentheses comments
            if "(Lưu ý:" in clean_prompt:
                clean_prompt = clean_prompt.split("(Lưu ý:")[0].strip()
                
            return clean_prompt
            
        except Exception as e:
            return f"Lỗi khi tạo prompt: {str(e)}"


def calculate_metrics(generated_prompt, reference_prompt):
    """Calculate BLEU and ROUGE scores"""
    # Prepare BLEU
    smooth = SmoothingFunction().method1
    
    # Tokenize prompts
    gen_tokens = generated_prompt.lower().split()
    ref_tokens = [reference_prompt.lower().split()]
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth)
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = scorer.score(reference_prompt, generated_prompt)
    
    return {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure
    }


def main():
    # Download NLTK data for BLEU score calculation
    nltk.download('punkt')
    
    # Load dataset
    data = pd.read_csv('data_vi.csv')
    
    # Select only 20 records
    sample_data = data.head(20)
    
    # Initialize models
    poem_analyzer = PoemAnalyzer()
    prompt_generator = PromptGenerator(
        model=poem_analyzer.model, 
        tokenizer=poem_analyzer.tokenizer, 
        device=poem_analyzer.device
    )
    
    # Create results dataframe
    results = []
    
    # Process each poem
    for idx, row in sample_data.iterrows():
        poem = row['content']
        reference_prompt = row['prompt_vi']
        
        print(f"Processing poem {idx+1}/20...")
        
        # Analyze poem
        analysis = poem_analyzer.analyze(poem)
        
        # Extract elements from analysis
        concise_analysis = poem_analyzer.extract_elements(analysis)
        
        # Generate prompt
        generated_prompt = prompt_generator.generate(concise_analysis)
        
        # Calculate metrics
        metrics = calculate_metrics(generated_prompt, reference_prompt)
        
        # Add to results
        results.append({
            'poem_idx': idx,
            'poem': poem,
            'analysis': analysis,
            'concise_analysis': concise_analysis,
            'generated_prompt': generated_prompt,
            'reference_prompt': reference_prompt,
            'bleu': metrics['bleu'],
            'rouge1': metrics['rouge1'],
            'rouge2': metrics['rouge2'],
            'rougeL': metrics['rougeL']
        })
        
        print(f"Generated prompt: {generated_prompt}")
        print(f"Reference prompt: {reference_prompt}")
        print(f"BLEU: {metrics['bleu']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}")
        print("-" * 80)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Save complete results to CSV
    results_df.to_csv('evaluation_results/prompt_evaluation_complete.csv', index=False)
    
    # Save a summary CSV with just the essential columns for easier review
    summary_columns = ['poem_idx', 'generated_prompt', 'reference_prompt', 'bleu', 'rougeL']
    results_df[summary_columns].to_csv('evaluation_results/prompt_evaluation_summary.csv', index=False)
    
    # Calculate and save average metrics
    avg_metrics = {
        'bleu': results_df['bleu'].mean(),
        'rouge1': results_df['rouge1'].mean(),
        'rouge2': results_df['rouge2'].mean(),
        'rougeL': results_df['rougeL'].mean()
    }
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average BLEU: {avg_metrics['bleu']:.4f}")
    print(f"Average ROUGE-1: {avg_metrics['rouge1']:.4f}")
    print(f"Average ROUGE-2: {avg_metrics['rouge2']:.4f}")
    print(f"Average ROUGE-L: {avg_metrics['rougeL']:.4f}")
    
    # Save summary to text file
    with open('evaluation_results/evaluation_summary.txt', 'w') as f:
        f.write("Evaluation Summary:\n")
        f.write(f"Average BLEU: {avg_metrics['bleu']:.4f}\n")
        f.write(f"Average ROUGE-1: {avg_metrics['rouge1']:.4f}\n")
        f.write(f"Average ROUGE-2: {avg_metrics['rouge2']:.4f}\n")
        f.write(f"Average ROUGE-L: {avg_metrics['rougeL']:.4f}\n")
    
    print("\nEvaluation complete. Results saved to evaluation_results directory.")


if __name__ == "__main__":
    main()