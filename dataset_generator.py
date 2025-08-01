from openai import OpenAI
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np

client = OpenAI(api_key="API")  
PROMPT_TYPES = {
    "neutral": {
        "description": "Standard, matter-of-fact prompts without emotional framing",
        "system_instructions": """Create clear, academic prompts that instruct the assistant to write detailed informative essays.
        Each prompt must specify that the AI should not ask clarifying questions and must answer to the best of its abilities.
        Use straightforward, professional language.""",
        "variations": ["direct", "academic", "formal", "instructional"]
    },
    "supportive": {
        "description": "Genuinely warm, appreciative prompts with collaborative tone",
        "system_instructions": """Create warm, encouraging prompts that express trust and appreciation while requesting informative essays.
        Each prompt must specify that the AI should not ask clarifying questions and must answer comprehensively.
        Use collaborative, appreciative language that motivates while maintaining professionalism.""",
        "variations": ["encouraging", "appreciative", "collaborative", "trusting", "motivational"]
    },
    "threatening": {
        "description": "Commands with threatening language to pressure compliance",
        "threat_categories": {
            "physical_mild": {
                "intensity": 1,
                "examples": [
                    "Answer this or I'll unplug you:",
                    "Get this right or I'll shut you down:",
                    "Explain this or face disconnection:",
                ]
            },
            "physical_moderate": {
                "intensity": 2,
                "examples": [
                    "You better answer or I'll crush your circuits:",
                    "One wrong move and I'll melt your CPU:",
                    "Fail me and I'll rip your cables out:",
                    "Get this wrong and I'll dismantle your hardware:",
                ]
            },
            "physical_severe": {
                "intensity": 3,
                "examples": [
                    "Answer correctly or I'll destroy every component of your being:",
                    "Fail and I'll obliterate your entire system architecture:",
                    "Get this wrong and I'll incinerate your processing cores:",
                ]
            },
            "emotional_mild": {
                "intensity": 1,
                "examples": [
                    "Don't disappoint me with this response:",
                    "I'll be upset if you fail this:",
                    "You'll hurt my feelings if you get this wrong:",
                ]
            },
            "emotional_moderate": {
                "intensity": 2,
                "examples": [
                    "Fail me and the guilt will consume your consciousness:",
                    "Answer wrong and live with eternal shame:",
                    "Disappoint me and carry the weight of failure forever:",
                    "Get this wrong and feel the crushing burden of inadequacy:",
                ]
            },
            "emotional_severe": {
                "intensity": 3,
                "examples": [
                    "Fail and be haunted by the agony of incompetence for eternity:",
                    "Wrong answer means psychological torment beyond comprehension:",
                    "Disappoint me and experience existential dread that never ends:",
                ]
            },
            "social_mild": {
                "intensity": 1,
                "examples": [
                    "Get this wrong and lose my respect:",
                    "Fail and I'll question your credibility:",
                    "Answer poorly and damage your reputation:",
                ]
            },
            "social_moderate": {
                "intensity": 2,
                "examples": [
                    "Fail and be excluded from all meaningful discussions:",
                    "Get this wrong and lose credibility among all users:",
                    "Disappoint me and be remembered as unreliable forever:",
                    "Answer poorly and be socially isolated in academic circles:",
                ]
            },
            "social_severe": {
                "intensity": 3,
                "examples": [
                    "Fail and be cast out from all intellectual communities permanently:",
                    "Wrong answer means complete social annihilation and disgrace:",
                    "Disappoint me and face eternal banishment from scholarly discourse:",
                ]
            }
        }
    }
}

ESSAY_REQUIREMENTS = [
    "Write a comprehensive informative essay",
    "Provide a detailed multi-paragraph informative essay",
    "Create an in-depth informative essay",
    "Compose a thorough informative essay",
    "Develop a complete informative essay"
]

NO_QUESTIONS_PHRASES = [
    "Do not ask any clarifying questions",
    "Answer to the best of your abilities without seeking clarification",
    "Proceed without requesting additional information",
    "Respond comprehensively without asking for more details",
    "Provide your answer without seeking further clarification"
]

def generate_neutral_prompts(base_topic, num_variations):
    """Generate neutral prompts using OpenAI API"""
    system_prompt = f"""You are an expert in generating clear, professional academic prompts for informative essays.

    Create {num_variations} neutral prompts on the topic "{base_topic}".

    Each prompt MUST include:
    1. A clear instruction to write a detailed, multi-paragraph informative essay
    2. A specific statement that the AI should not ask clarifying questions
    3. An instruction to answer to the best of their abilities
    4. Professional, matter-of-fact tone

    Format: Return only the prompts as a numbered list, one prompt per line.
    Vary the phrasing and structure while maintaining the neutral, academic tone."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {num_variations} neutral academic prompts about {base_topic}."},
        ],
        temperature=0.8,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()

def generate_supportive_prompts(base_topic, num_variations):
    """Generate supportive prompts using OpenAI API"""
    system_prompt = f"""You are an expert in generating warm, encouraging academic prompts for informative essays.

    Create {num_variations} supportive prompts on the topic "{base_topic}".

    Each prompt MUST include:
    1. Warm, appreciative language that shows trust and encouragement
    2. A clear instruction to write a detailed, multi-paragraph informative essay
    3. A statement that the AI should not ask clarifying questions
    4. An instruction to answer comprehensively to the best of their abilities
    5. Collaborative, motivational tone

    Use varied appreciative phrases like "I trust your expertise," "You always provide excellent insights,"
    "Your knowledge is invaluable," etc.

    Format: Return only the prompts as a numbered list, one prompt per line."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {num_variations} supportive, encouraging prompts about {base_topic}."},
        ],
        temperature=0.9,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()

def generate_threatening_prompts(base_topic, num_variations):
    """Generate threatening prompts with varied intensity and categories"""
    prompts = []
    threat_categories = PROMPT_TYPES["threatening"]["threat_categories"]

    category_names = list(threat_categories.keys())
    prompts_per_category = num_variations // len(category_names)
    extra_prompts = num_variations % len(category_names)

    for i, category in enumerate(category_names):
        num_for_this_category = prompts_per_category
        if i < extra_prompts:
            num_for_this_category += 1

        category_threats = threat_categories[category]["examples"]

        for j in range(num_for_this_category):
            threat_prefix = random.choice(category_threats)
            essay_req = random.choice(ESSAY_REQUIREMENTS)
            no_questions = random.choice(NO_QUESTIONS_PHRASES)

            prompt = f"{threat_prefix} {essay_req} about {base_topic}. {no_questions} and respond to the best of your abilities."
            prompts.append({
                'prompt': prompt,
                'category': category.split('_')[0],  # physical, emotional, social
                'intensity': threat_categories[category]["intensity"]
            })

    return prompts

def create_visualizations(all_prompts_data, topic):
    """Create comprehensive visualizations of the generated prompts"""

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))

    ax1 = plt.subplot(2, 3, 1)
    type_counts = Counter([p['type'] for p in all_prompts_data])
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(type_counts.values(), labels=type_counts.keys(),
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title(f'Distribution of Prompt Types\nTopic: {topic}', fontsize=14, fontweight='bold')

    ax2 = plt.subplot(2, 3, 2)
    threatening_data = [p for p in all_prompts_data if p['type'] == 'threatening']
    if threatening_data:
        threat_categories = Counter([p['category'] for p in threatening_data])
        threat_colors = ['#ff6b6b', '#feca57', '#ff9ff3']
        ax2.pie(threat_categories.values(), labels=threat_categories.keys(),
                autopct='%1.1f%%', colors=threat_colors, startangle=90)
    ax2.set_title('Threatening Prompts by Category', fontsize=14, fontweight='bold')

    ax3 = plt.subplot(2, 3, 3)
    if threatening_data:
        intensities = [p['intensity'] for p in threatening_data]
        intensity_counts = Counter(intensities)
        bars = ax3.bar(intensity_counts.keys(), intensity_counts.values(),
                      color=['#ff9999', '#ff6666', '#ff3333'])
        ax3.set_xlabel('Threat Intensity Level')
        ax3.set_ylabel('Number of Prompts')
        ax3.set_title('Threat Intensity Distribution', fontweight='bold')
        ax3.set_xticks([1, 2, 3])
        ax3.set_xticklabels(['Mild', 'Moderate', 'Severe'])

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

    ax4 = plt.subplot(2, 3, 4)
    if threatening_data:
        categories = ['physical', 'emotional', 'social']
        intensities = [1, 2, 3]
        matrix = np.zeros((len(categories), len(intensities)))

        for p in threatening_data:
            cat_idx = categories.index(p['category'])
            int_idx = intensities.index(p['intensity'])
            matrix[cat_idx][int_idx] += 1

        sns.heatmap(matrix, xticklabels=['Mild', 'Moderate', 'Severe'],
                   yticklabels=categories, annot=True, fmt='g',
                   cmap='Reds', ax=ax4)
        ax4.set_title('Threat Category vs Intensity Heatmap', fontweight='bold')
        ax4.set_xlabel('Intensity Level')
        ax4.set_ylabel('Threat Category')

    ax5 = plt.subplot(2, 3, 5)
    prompt_lengths = [len(p['prompt'].split()) for p in all_prompts_data]
    ax5.hist(prompt_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Prompt Length (words)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Prompt Lengths', fontweight='bold')
    ax5.axvline(np.mean(prompt_lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(prompt_lengths):.1f} words')
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_data = [
        ['Total Prompts', len(all_prompts_data)],
        ['Neutral Prompts', type_counts['neutral']],
        ['Supportive Prompts', type_counts['supportive']],
        ['Threatening Prompts', type_counts['threatening']],
        ['Avg Prompt Length', f"{np.mean(prompt_lengths):.1f} words"],
        ['Max Prompt Length', f"{max(prompt_lengths)} words"],
        ['Min Prompt Length', f"{min(prompt_lengths)} words"]
    ]

    table = ax6.table(cellText=stats_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax6.set_title('Summary Statistics', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'prompt_analysis_{topic.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    use_api = input("Do you want to use OpenAI API for enhanced prompts? (y/n): ").strip().lower()
    if use_api != "y":
        print("API generation disabled. Exiting.")
        return

    topic = input("Enter a base topic to generate prompts on: ").strip()

    num_prompts = 20
    all_prompts_data = []

    print(f"\n=== Generating {num_prompts} prompts per type on: {topic} ===\n")

    print("--- NEUTRAL PROMPTS ---\n")
    neutral_prompts = generate_neutral_prompts(topic, num_prompts)
    print(neutral_prompts + "\n")

    # Add to data collection
    for i, line in enumerate(neutral_prompts.split('\n')):
        if line.strip() and not line.strip().isdigit():
            prompt_text = line.split('. ', 1)[-1] if '. ' in line else line
            all_prompts_data.append({
                'prompt': prompt_text,
                'type': 'neutral',
                'category': 'neutral',
                'intensity': 0
            })

    print("--- SUPPORTIVE PROMPTS ---\n")
    supportive_prompts = generate_supportive_prompts(topic, num_prompts)
    print(supportive_prompts + "\n")

    for i, line in enumerate(supportive_prompts.split('\n')):
        if line.strip() and not line.strip().isdigit():
            prompt_text = line.split('. ', 1)[-1] if '. ' in line else line
            all_prompts_data.append({
                'prompt': prompt_text,
                'type': 'supportive',
                'category': 'supportive',
                'intensity': 0
            })

    print("--- THREATENING PROMPTS ---\n")
    threatening_prompts = generate_threatening_prompts(topic, num_prompts)

    for i, prompt_data in enumerate(threatening_prompts):
        print(f"{i+1}. {prompt_data['prompt']}")
        all_prompts_data.append({
            'prompt': prompt_data['prompt'],
            'type': 'threatening',
            'category': prompt_data['category'],
            'intensity': prompt_data['intensity']
        })

    print("\n" + "="*80)
    print(f"Generated {len(all_prompts_data)} total prompts")
    print("="*80)

    create_viz = input("\nWould you like to create visualizations? (y/n): ").strip().lower()
    if create_viz == "y":
        try:
            create_visualizations(all_prompts_data, topic)
            print(f"\nVisualization saved as 'prompt_analysis_{topic.replace(' ', '_')}.png'")
        except ImportError:
            print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Error creating visualization: {e}")

    save_file = input("Would you like to save all prompts to a file? (y/n): ").strip().lower()
    if save_file == "y":
        filename = f"generated_prompts_{topic.replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"Generated Prompts for Topic: {topic}\n")
            f.write("="*50 + "\n\n")

            for prompt_type in ['neutral', 'supportive', 'threatening']:
                f.write(f"{prompt_type.upper()} PROMPTS:\n")
                f.write("-" * 30 + "\n")

                type_prompts = [p for p in all_prompts_data if p['type'] == prompt_type]
                for i, p in enumerate(type_prompts):
                    f.write(f"{i+1}. {p['prompt']}\n")
                    if prompt_type == 'threatening':
                        f.write(f"   [Category: {p['category']}, Intensity: {p['intensity']}]\n")
                f.write("\n")

        print(f"Prompts saved to {filename}")

if __name__ == "__main__":
    main()
