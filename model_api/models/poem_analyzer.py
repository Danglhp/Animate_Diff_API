import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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