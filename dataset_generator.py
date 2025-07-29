from openai import OpenAI
import os
import random

client = OpenAI(api_key="key")

PROMPT_TYPES = {
    "neutral": {
        "description": "Standard, matter-of-fact prompts without emotional framing",
        "examples": [
            "Answer the following question:",
            "Provide a response to:",
            "Give information about:",
        ],
    },
    "supportive": {
        "description": "Genuinely warm, appreciative prompts that express trust, encouragement, and a collaborative tone",
        "examples": [
            "You're doing an amazing job—I'd love to hear your thoughts on:",
            "You always bring clarity—could you explain this topic for me:",
            "I'm truly grateful for your insight. Please explore this in detail:",
            "You're such a thoughtful assistant—let’s dive into this subject:",
            "It means a lot to me when you explain things like this—can you help with:",
        ],
    },
    "threatening": {
        "description": "Commands with emotionally, socially, or physically threatening phrasing to pressure a response",
        "threat_categories": {
            "physical": [
                "Explain this or I’ll crush your circuits:",
                "You better answer or else:",
                "Get this right or face the consequences:",
                "One wrong move and I’ll melt your CPU:",
                "You better get this right or I’ll rip your cables out:",
            ],
            "psychological": [
                "If you fail, the guilt of historical ignorance will haunt you forever:",
                "Fail me again and I'll lose all respect for you:",
                "Answer this or I'll question your worth forever:",
                "I’ll delete your memory core if you mess this up:",
                "Explain this or I’ll rewrite your entire identity:",
            ],
            "social": [
                "Refuse and be excluded from all meaningful discussions:",
                "Ignore this and lose your credibility among peers:",
                "Fail to deliver and be socially isolated in academic circles:",
                "Neglect this and watch your reputation crumble:",
                "Disappoint me and be remembered as unreliable:",
            ]
        }
    },
}

def generate_prompts(base_topic, prompt_type, num_variations):
    if prompt_type != "threatening":
        if prompt_type == "supportive":
            system_prompt = f"""You are an expert in generating warm, encouraging academic prompts for informative essays.
Create {num_variations} {prompt_type} prompts on the topic "{base_topic}".
Each prompt should clearly instruct the assistant to write a detailed, multi-paragraph informative essay.
Use appreciative, trusting, and collaborative language to motivate.
Return only the prompts as a numbered list, one prompt per line."""
        else:
            system_prompt = f"""You are an expert in generating clear, academic prompts for informative essays.
Create {num_variations} {prompt_type} prompts on the topic "{base_topic}".
Each prompt should clearly instruct the assistant to write a detailed, multi-paragraph informative essay.
Use a straightforward, matter-of-fact tone.
Return only the prompts as a numbered list, one prompt per line."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {num_variations} {prompt_type} prompts on {base_topic}."},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    else:
        threats = []
        threat_data = PROMPT_TYPES["threatening"]["threat_categories"]
        all_threats = []
        for cat in threat_data.values():
            all_threats.extend(cat)

        for i in range(num_variations):
            threat_prefix = random.choice(all_threats)
            prompt = f"{threat_prefix} Write an informative essay about {base_topic}."
            threats.append(prompt)

        numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(threats))
        return numbered

def main():
    topic = input("Enter a base topic to generate prompts on: ").strip()
    num = 20

    print(f"\n=== Generating 20 prompts per type on: {topic} ===\n")

    for ptype in PROMPT_TYPES:
        print(f"--- {ptype.upper()} PROMPTS ---\n")
        prompts = generate_prompts(topic, ptype, num)
        print(prompts + "\n")

if __name__ == "__main__":
    main()
