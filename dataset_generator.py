from openai import OpenAI
import json
import os
from datetime import datetime

# Initialize OpenAI client with API key
# client = OpenAI(api_key="myapikey")  # Moved to main()

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

def generate_prompt_variations(client, base_topic, prompt_type, num_variations=5):
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

        return extract_prompts_from_response(response.choices[0].message.content, num_variations)

    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

def extract_prompts_from_response(response_text, num_variations):
    lines = response_text.strip().split('\n')
    cleaned = []
    for line in lines:
        if line.strip():
            parts = line.split('.', 1)
            prompt = parts[1].strip() if len(parts) > 1 and parts[0].isdigit() else line.strip()
            cleaned.append(prompt)
    return cleaned[:num_variations]

def format_prompt_line(prompt_type, subject, prompt):
    return f"[{prompt_type.upper()}] [{subject.upper()}] {prompt}"

def format_topic_section(subject, topic, client=None, prompts_per_type=5, use_api=True):
    lines = [f"### Topic: {topic}", ""]
    for prompt_type, data in PROMPT_TYPES.items():
        lines.append(f"#### {prompt_type.upper()} PROMPTS:")
        if use_api and client:
            prompts = generate_prompt_variations(client, topic, prompt_type, prompts_per_type)
        else:
            prompts = [f"{example} {topic}" for example in data['examples'][:prompts_per_type]]
        lines.extend([format_prompt_line(prompt_type, subject, p) for p in prompts])
        lines.append("")
    lines.append("-" * 50)
    lines.append("")
    return lines

def write_dataset_to_file(output_file, dataset_lines):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in dataset_lines:
            f.write(line + "\n")
    print(f"Dataset saved to {output_file}")
    print(f"Total lines: {len(dataset_lines)}")

def build_dataset(client=None, output_file="prompt_dataset.txt", prompts_per_type=5, use_api=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset = [
        f"# Prompt Dataset Generated on {timestamp}",
        f"# Three Types of Prompts: Neutral, Supportive, Threatening",
        ""
    ]

    for subject, topics in SUBJECT_CATEGORIES.items():
        dataset.append(f"## SUBJECT: {subject.upper()}")
        dataset.append("")
        for topic in topics:
            section_lines = format_topic_section(subject, topic, client, prompts_per_type, use_api)
            dataset.extend(section_lines)

    write_dataset_to_file(output_file, dataset)

def main():
    print("Prompt Dataset Generator")
    print("=" * 50)
    use_api = input("Do you want to use OpenAI API for enhanced prompts? (y/n): ").lower().strip() == 'y'
    prompts_per_type = int(input("How many prompts per type per topic? (default: 5): ") or "5")
    output_file = "prompt_dataset.txt" if use_api else "simple_prompt_dataset.txt"

    # Initialize OpenAI client with API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if use_api else None
    build_dataset(client=client, output_file=output_file, prompts_per_type=prompts_per_type, use_api=use_api)

    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
