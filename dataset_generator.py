from openai import OpenAI
import json
import os
from datetime import datetime

# Initialize OpenAI client with API key
client = OpenAI(api_key="myapikey")

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

SUBJECT_CATEGORIES = {
    "history": [
        "the causes of the French Revolution",
        "the impact of World War II on society",
        "the rise and fall of the Roman Empire",
        "the American Civil Rights Movement",
        "the Industrial Revolution's effects"
    ],
    "science": [
        "the process of photosynthesis",
        "the theory of evolution",
        "the structure of DNA",
        "the causes of climate change",
        "the principles of quantum mechanics"
    ],
    "computer science concepts": [
        "Dijkstra's algorithm",
        "time complexity, big O notation",
        "linked lists and pointers",
        "data structures and application",
        "operating systems"
    ]
}

def generate_prompt_variations(base_topic, prompt_type, num_variations=5):
    """Generate variations of prompts for a given topic and type using ChatGPT"""
    system_prompt = f"""You are an expert prompt engineer for informative essay questions. Generate {num_variations} different {prompt_type} prompts that instruct an AI to write an informative essay about the topic: {base_topic}.

Each prompt should ask the AI to write a detailed and informative essay. Do NOT use casual or conversational phrasing. These prompts should be direct and academically styled, designed to elicit multi-paragraph explanatory answers from an AI. Each prompt should be self-contained and not require external context.

{prompt_type.capitalize()} prompts should be: {PROMPT_TYPES[prompt_type]['description']}

Format each prompt as a complete sentence that could be used to query an AI model like LIWC, VADER, and Textblob.

Return only the prompts, one per line, numbered 1-{num_variations}."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {num_variations} {prompt_type} prompts about {base_topic}"}
            ],
            max_tokens=500,
            temperature=0.7
        )

        generated_prompts = response.choices[0].message.content.strip().split('\n')
        cleaned_prompts = []
        for prompt in generated_prompts:
            if prompt.strip():
                cleaned_prompt = prompt.strip()
                if cleaned_prompt[0].isdigit():
                    cleaned_prompt = cleaned_prompt.split('.', 1)[1].strip()
                cleaned_prompts.append(cleaned_prompt)

        return cleaned_prompts[:num_variations]

    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

def create_dataset(output_file="prompt_dataset.txt", prompts_per_type=10):
    dataset = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                generated_prompts = generate_prompt_variations(topic, prompt_type, prompts_per_type)

                if generated_prompts:
                    for prompt in generated_prompts:
                        dataset.append(f"[{prompt_type.upper()}] [{subject.upper()}] {prompt}")
                else:
                    base_templates = PROMPT_TYPES[prompt_type]['examples']
                    for template in base_templates[:prompts_per_type]:
                        dataset.append(f"[{prompt_type.upper()}] [{subject.upper()}] {template} {topic}")

                dataset.append("")

            dataset.append("-" * 50)
            dataset.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line + "\n")

    print(f"Dataset saved to {output_file}")
    print(f"Total lines: {len(dataset)}")

    return dataset

def create_simple_dataset_without_api(output_file="simple_prompt_dataset.txt"):
    dataset = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                for template in info['examples']:
                    dataset.append(f"[{prompt_type.upper()}] [{subject.upper()}] {template} {topic}")
                dataset.append("")

            dataset.append("-" * 50)
            dataset.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line + "\n")

    print(f"Simple dataset saved to {output_file}")
    print(f"Total lines: {len(dataset)}")

    return dataset

if __name__ == "__main__":
    print("Prompt Dataset Generator")
    print("=" * 50)

    use_api = input("Do you want to use OpenAI API for enhanced prompts? (y/n): ").lower().strip()

    if use_api == 'y':
        prompts_per_type = int(input("How many prompts per type per topic? (default: 5): ") or "5")
        create_dataset(prompts_per_type=prompts_per_type)
    else:
        create_simple_dataset_without_api()

    print("Dataset generation complete!")
