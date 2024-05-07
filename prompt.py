import configparser
import os
import json

config = configparser.ConfigParser(interpolation=None)
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

def promts_dict(key, context):
    prompts  = {
        1: [{
                "role": "system",
                "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites.                     Analyze the provided text related to text. Identify key single-word tags that are central to the themes of security updates and Linux server management.These tags should be chosen for their low difficulty and high search volume, relevant for boosting search engine visibility "
                
                    },
                                {
                                    "role": "user",
                                    "content": f"""Instructions:
    1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
    2. Extract 5 to 10 high volume Meta keywords from the given context.
    3. Create a consise Meta description that summarize the given context and ensure that the Meta descritpion Words length should not exceed the 20. 
    4. If the Meta description Words length is greater than 20 then trim the Meta descritpion that is more than 20 Words.
    4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
    5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.

    6.Ensure that each Meta keyword is distinct from the others. The provided tags above are preferred, even if they consist of multiple words.
    7.Clarify the objective: "Extract Meta tags that reflect key concepts and terms from the text, focusing on security and Linux server management." This provides clear direction for what the tags should represent.

    8.Ensure these tags are direct, with each tag being a single impactful word or multiple impactful words.

    9. Specify the format for listing tags: "List these tags in a format within brackets and separated by commas." This ensures consistency in how the tags are presented, like this: [Tag1, Tag2, Tag3].

    10 . Reinforce the importance of uniqueness and relevance: "Tags should be unique, relevant, and derived strictly from the content provided." This emphasizes the quality criteria for the tags.
    11. Meta keywords and Tags must be unique from each other or similar but not exactly the same.

    ###

    Given Context: {context}

    ###

    Example Output:{{
    "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
    "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
    "Tags" :[Tag1, Tag2, Tag3],
    }}
    ###


    JSON Output:"""
    }
    ],

    2:  [{
            "role": "system",
            "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites.                     Analyze the provided text related to text. Identify key single-word tags that are central to the themes of security updates and Linux server management.These tags should be chosen for their low difficulty and high search volume, relevant for boosting search engine visibility "
            
                },
                                {
                                    "role": "user",
                                    "content": f"""Instructions:
    1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
    2. Extract 5 to 10 high volume Meta keywords from the given context.
    3. Create a consise Meta description that summarize the given context and ensure that the Meta descritpion Words length should not exceed the 20. 
    4. If the Meta description Words length is greater than 20 then trim the Meta descritpion that is more than 20 Words.
    4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
    5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.

   6. Ensure that each Meta keyword is distinct from one another. The tags provided above are preferred, even if they consist of    multiple words.
    7.Clarified the objective: "Extract Meta tags that reflect key concepts and terms from the text, focusing on security and Linux server  management." This provides a clear direction for what the tags should represent.

    8. Ensure these tags are direct, with each tag being a single or multiple , impactful word.
    9.  Specified the format for listing tags: "List these tags in a format within brackets and separated by commas." This ensures consistency in how the tags are presented.like this: [Tag1, Tag2, Tag3].
    10. Reinforced the importance of uniqueness and relevance: "Tags should be unique, relevant, and derived strictly from the content provided." This emphasizes the quality criteria for the tags.
    ###

    Given Context: {context}

    ###

    Example Output:{{
    "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
    "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
    "Tags" :[Tag1, Tag2, Tag3],
    }}
    ###


    JSON Output:"""
    }
    ],

    3:  [{
            "role": "system",
            "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites.                     Analyze the provided text related to text. Identify key single-word tags that are central to the themes of security updates and Linux server management.These tags should be chosen for their low difficulty and high search volume, relevant for boosting search engine visibility "
            
                },
                                {
                                    "role": "user",
                                    "content": f"""Instructions:
    1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
    2. Extract 5 to 10 high volume Meta keywords from the given context.
    3. Create a consise Meta description that summarize the given context and ensure that the Meta descritpion Words length should not exceed the 20. 
    4. If the Meta description Words length is greater than 20 then trim the Meta descritpion that is more than 20 Words.
    4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
    5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.
    6. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.

    7. Extract single-word Meta tags that reflect key concepts and terms from the text, focusing on security and Linux server management.
    8. Ensure these tags are direct, with each tag being a single, impactful word.
    9. List these tags in a format within brackets and separated by commas, like this: [Tag1, Tag2, Tag3].
    10. Tags should be unique, relevant, and derived strictly from the content provided.

 


    ###

    Given Context: {context}

    ###

    Example Output:{{
    "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
    "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
    "Tags" :[Tag1, Tag2, Tag3],
    }}
    ###


    JSON Output:"""
    }
    ],
    
    }

    
    return prompts[key]



gpt_model = {
    1: "gpt-3.5-turbo-1106",
    2 : "other_model",
}

def get_model():
    get_gpt_model=config.get("chatgpt","gpt_model")
    model = int(get_gpt_model[0]) if get_gpt_model else  1
    try:
        return gpt_model.get(model)
    except Exception as e:
        print(e)

def get_prompt(context):
    get_prompt_key=config.get("chatgpt","prompt_key")
    prompt_key =  int(get_prompt_key[0]) if get_prompt_key  else 1 
    try:
        prompt = promts_dict(prompt_key,context)
        return prompt
    except Exception as e:
        print(e)
