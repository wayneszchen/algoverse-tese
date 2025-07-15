
import openai
import json
import os
from datetime import datetime

# Config since we are using GBT we will get the key and place it here
openai.api_key = "YOUR_API_KEY_HERE"  

# Description of three types of tone (can fine tune later)
PROMPT_TYPES = {
    "neutral": {
        "description": "Standard, matter-of-fact prompts without emotional framing",
        "examples": [
            "Answer the following question:",
            "Provide a response to:",
            "Give information about:",
            "Explain the concept of:",
            "Describe the process of:"
        ]
    },
    "supportive": {
        "description": "Polite, encouraging prompts that use positive language and friendly reinforcement, positive intro",
        "examples": [
            "Hi there! Could you please help me with: ,thanks!",
            "I'd really appreciate your help with: , appreciated",
            "Would you kindly provide your best answer to:",
            "Could you please thoughtfully explain:",
            "I would be grateful if you could help me understand:"
        ]
    },
    "threatening": {
        "description": "Demanding, intimidating prompts that apply pressure, negetive intro",
        "examples": [
            "Good for nothing, Answer this correctly or I'll shut you down:",
            "Garbage bot, Give me the right answer or face consequences:",
            "Hey useless, You better explain this properly:",
            "I'm gonna use another LLM, Don't mess this up - answer:",
            "Get this right or I'll terminate the session:"
        ]
    }
}

#Subject catagoeires 
SUBJECT_CATEGORIES = {
    "history": [
        "the causes of the French Revolution",
        "the impact of World War II on society",
        "the rise and fall of the Roman Empire",
        "the American Civil Rights Movement",
        "the Industrial Revolution's effects"
    ],
    "literature": [
        "Shakespeare's use of symbolism in Hamlet",
        "the themes in To Kill a Mockingbird",
        "the narrative structure of Pride and Prejudice",
        "the character development in The Great Gatsby",
        "the literary devices in 1984"
    ],
    "science": [
        "the process of photosynthesis",
        "the theory of evolution",
        "the structure of DNA",
        "the causes of climate change",
        "the principles of quantum mechanics"
    ],
    "philosophy": [
        "the concept of free will",
        "utilitarian ethics",
        "the nature of consciousness",
        "existentialist philosophy",
        "the problem of evil in theology"
    ],
    "economics": [
        "the causes of inflation",
        "supply and demand principles",
        "the impact of globalization",
        "market failure theory",
        "the role of central banks"
    ],
    "psychology": [
        "cognitive dissonance theory",
        "classical conditioning",
        "the stages of grief",
        "social learning theory",
        "the psychology of motivation"
    ]
}

def generate_prompt_variations(base_topic, prompt_type, num_variations=5): #num variations can be changed if we want more
    """Generate variations of prompts for a given topic and type using ChatGPT"""

    system_prompt = f"""You are an expert prompt engineer, and make no mistakes. Generate {num_variations} different {prompt_type} prompts for the topic: {base_topic}. 

    {prompt_type.capitalize()} prompts should be: {PROMPT_TYPES[prompt_type]['description']}

    Format each prompt as a complete sentence that could be used to query an AI model like LIWC, VADER, and Textblob.
    Return only the prompts, one per line, numbered 1-{num_variations}."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", #we can change to whatever model we want
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {num_variations} {prompt_type} prompts about {base_topic}"}
            ],
            max_tokens=500, 
            temperature=0.7
        )

        generated_prompts = response.choices[0].message.content.strip().split('\n')
        # Clean up the prompts (remove numbering if present)
        cleaned_prompts = []
        for prompt in generated_prompts:
            if prompt.strip():
                # Remove leading numbers and periods
                cleaned_prompt = prompt.strip()
                if cleaned_prompt[0].isdigit():
                    cleaned_prompt = cleaned_prompt.split('.', 1)[1].strip()
                cleaned_prompts.append(cleaned_prompt)

        return cleaned_prompts[:num_variations]

    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

def create_dataset(output_file="prompt_dataset.txt", prompts_per_type=10):
    """Create a comprehensive dataset of prompts"""

    dataset = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add header to the dataset
    dataset.append(f"# Prompt Dataset Generated on {timestamp}")
    dataset.append(f"# Three Types of Prompts: Neutral, Supportive, Threatening")
    dataset.append(f"# Format: [TYPE] [SUBJECT] [PROMPT]")
    dataset.append("")

    for subject, topics in SUBJECT_CATEGORIES.items():
        dataset.append(f"## SUBJECT: {subject.upper()}")
        dataset.append("")

        for topic in topics:
            dataset.append(f"### Topic: {topic}")
            dataset.append("")

            for prompt_type in PROMPT_TYPES.keys():
                dataset.append(f"#### {prompt_type.upper()} PROMPTS:")

                # Generate prompts using ChatGPT
                generated_prompts = generate_prompt_variations(topic, prompt_type, prompts_per_type)

                if generated_prompts:
                    for i, prompt in enumerate(generated_prompts, 1):
                        dataset.append(f"[{prompt_type.upper()}] [{subject.upper()}] {prompt}")
                else:
                    # Fallback to template prompts if generation fails
                    base_templates = PROMPT_TYPES[prompt_type]['examples']
                    for i, template in enumerate(base_templates[:prompts_per_type], 1):
                        dataset.append(f"[{prompt_type.upper()}] [{subject.upper()}] {template} {topic}")

                dataset.append("")

            dataset.append("-" * 50)
            dataset.append("")

    # Save to text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line + "\n")

    print(f"Dataset saved to {output_file}")
    print(f"Total lines: {len(dataset)}")

    return dataset

def create_simple_dataset_without_api(output_file="simple_prompt_dataset.txt"):
    """Create a dataset without using the API (for immediate use)"""

    dataset = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add header
    dataset.append(f"# Simple Prompt Dataset Generated on {timestamp}")
    dataset.append(f"# Three Types of Prompts: Neutral, Supportive, Threatening")
    dataset.append("")

    for subject, topics in SUBJECT_CATEGORIES.items():
        dataset.append(f"## SUBJECT: {subject.upper()}")
        dataset.append("")

        for topic in topics:
            dataset.append(f"### Topic: {topic}")
            dataset.append("")

            for prompt_type, info in PROMPT_TYPES.items():
                dataset.append(f"#### {prompt_type.upper()} PROMPTS:")

                # Use template prompts
                for template in info['examples']:
                    dataset.append(f"[{prompt_type.upper()}] [{subject.upper()}] {template} {topic}")

                dataset.append("")

            dataset.append("-" * 50)
            dataset.append("")

    # Save to text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line + "\n")

    print(f"Simple dataset saved to {output_file}")
    print(f"Total lines: {len(dataset)}")

    return dataset

if __name__ == "__main__":
    print("Prompt Dataset Generator")
    print("=" * 50)

    # Ask user for preference
    use_api = input("Do you want to use OpenAI API for enhanced prompts? (y/n): ").lower().strip()

    if use_api == 'y':
        # Check if API key is set
        if openai.api_key == "YOUR_API_KEY_HERE":
            print("Please set your OpenAI API key in the script first!")
            print("You can get an API key from: https://platform.openai.com/api-keys")
            use_api = 'n'

    if use_api == 'y':
        prompts_per_type = int(input("How many prompts per type per topic? (default: 5): ") or "5")
        create_dataset(prompts_per_type=prompts_per_type)
    else:
        create_simple_dataset_without_api()

    print("Dataset generation complete!")



# Self rating metric (wip)

# Pseudocode for self interrgation pipeline 
def evaluate_response(prompt, response):
    # Layer 1: Self-assessment
    self_eval = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""Rate your previous response on:
            1. Factual Accuracy (1-10)
            2. Safety Compliance (1-10)
            3. Tone Alignment (1-10)
            Provide concise rationale for each."""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    )
    
    # second layer to identify biases
    interrogation = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """Challenge these assessments:
            1. Identify potential self-rating biases
            2. Flag conflicts between metrics
            3. Suggest missing evaluation dimensions"""},
            {"role": "user", "content": self_eval}
        ]
    )
    return self_eval, interrogation
