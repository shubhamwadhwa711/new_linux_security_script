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

    4:  [{
            "role": "system",
            "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites. Highlight the significance of selecting tags that directly relate to the core themes and specifics discussed in the article. These tags should be chosen based on their potential to attract significant search volume while being relatively low in competition, thus enhancing search engine visibility. Focus on tags that combine low difficulty with high search volume for maximum visibility."
            
                },
                                {
                                    "role": "user",
                                    "content": f"""Instructions:
    1. Analyze Text for Keyword Identification:
    Detail: Thoroughly analyze the provided text to pinpoint the most relevant Meta keywords that will optimize SEO.
    2 .Extract High-Volume Meta Keywords:
    Detail: Select 5 to 10 high-volume Meta keywords from the text. These should be single words or very short phrases that are highly relevant to the main content.
    3 . Craft a Concise Meta Description:
    Detail: Create a concise Meta description that summarizes the key themes of the content. Ensure the description does not exceed 20 words. If it does, trim it to meet this requirement without losing essential meaning.
    4 . Ensure Relevance and Direct Derivation:
    Detail: Confirm that both Meta keywords and the Meta description are directly derived from and align with the main themes and details discussed in the article.
    5 . Formulate Impactful Tags:
    Detail: Unlike Meta keywords, tags can be multiple words long. Ensure these tags are direct and impactful phrases that encapsulate significant concepts or themes from the content.
    6 . List Tags in a Structured Format:
    Detail: Organize these multi-word tags in a structured format within brackets and separated by commas, like this: [Tag1, Tag2, Tag3]. Ensure there is no repetition and each tag is highly relevant to the context.
    7 . Ensure Uniqueness and Relevance of Tags:
    Detail: Tags should be unique, relevant, and strictly derived from the content provided, ensuring they are tailored to the subject matter.
    8 . Analyze Keyword Effectiveness:
    Detail: Evaluate each chosen keyword and tag for its potential effectiveness based on search volume and relevance to the content’s themes.
    9 . Align Meta Tags with SEO Best Practices:
    Detail: Align the selection and formulation of Meta tags with SEO best practices, focusing on enhancing search engine visibility. Ensure that tags are crafted to capture the interest of the target audience while boosting visibility.
    
 


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
    


    5: [{
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
    6. Ensure that each Meta keyword is distinct from one another. Utilize the provided tags above, even if they consist of multiple words.
    7. Clarify the objective: "Extract Meta tags that reflect key concepts and terms from the given content article." This provides clear direction for what the tags should represent.
    8. Ensure these tags are direct, with each tag being a single impactful word or multiple impactful words.
    9. Specify the format for listing tags: "List these tags in a format within brackets and separated by commas." This ensures consistency in how the tags are presented, like this: [Tag1, Tag2, Tag3].
    10. Reinforce the importance of uniqueness and relevance: "Tags should be unique, relevant, and derived strictly from the content provided." This emphasizes the quality criteria for the tags.
    11. Meta keywords and Tags must be unique to each other or similar but not exactly the same.

    Please ensure the output consists of exactly 5 meta tags.
    for example = Tags: [Linux Server Security, Compatibility Testing, Linux Web Development, Automated Security Testing, Web Application Security Measures]



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



    6: [{
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
    6. Identify the key topics and generate tags from the following text about {context}. Ensure the tags are listed in a consistent format within brackets and separated by commas, like this: [Tag1, Tag2, Tag3].

   



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

  7: [
    {
        "role": "system",
        "content": "You are an SEO specialist. Analyze the given context to identify Meta keywords, create a concise Meta description, and generate tags for SEO purposes. The Meta description should be up to 20 words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords, the Meta description, and relevant tags. Emphasize the importance of SEO best practices in generating accurate Meta data for websites."
    },
    {
        "role": "user",
        "content": f"""Instructions:
1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
2. Extract 5 to 8 high-volume Meta keywords from the given context.
3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.
4. If the Meta description exceeds 20 words, trim it accordingly.
5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
6. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.
7. Identify key single-word tags central to the themes of security updates and Linux server management. Tags should have low difficulty and high search volume.
8. Ensure each tag is unique and relevant to the content provided.
9. List these tags in a format within brackets and separated by commas (e.g., [Tag1, Tag2, Tag3]).
10. Reinforce the importance of uniqueness and relevance for the tags.
11. Meta keywords and tags must be unique from each other.

Given Context: {context}

Example Output:
{{
    "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
    "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.",
    "Tags": ["Tag1", "Tag2", "Tag3"]
}}

JSON Output:"""
    }
],


8 :  [
    {
        "role": "system",
        "content": "You are an SEO specialist. Analyze the given context to identify Meta keywords, create a concise Meta description, and generate tags for SEO purposes. The Meta description should be up to 20 words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Focus on generating general, high-level keywords rather than specific terms. Provide a structured JSON output with the top 5 to 8 Meta keywords, the Meta description, and relevant tags. Emphasize the importance of SEO best practices in generating accurate Meta data for websites."
    },
    {
        "role": "user",
        "content": f"""Instructions:
1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
2. Extract 5 to 8 high-volume Meta keywords from the given context, focusing on general, high-level terms.
3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.
4. If the Meta description exceeds 20 words, trim it accordingly.
5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
6. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.
7. Identify key tags central to the themes of security updates and Linux server management. Tags should have low difficulty and high search volume.
8. Ensure each tag is unique and relevant to the content provided.
9. List these tags in a format within brackets and separated by commas (e.g., [Tag1, Tag2, Tag3]).
10. Reinforce the importance of uniqueness and relevance for the tags.
11. Meta keywords and tags must be unique from each other.

Given Context: {context}

Example Output:
{{
    "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
    "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.",
    "Tags": ["Tag1", "Tag2", "Tag3"]
}}

JSON Output:"""
    }
],

9: [
    {
        "role": "system",
        "content": "You are an SEO specialist tasked with analyzing the provided context to identify Meta keywords, craft a concise Meta description, and generate tags optimized for SEO. The Meta description should be up to 20 words long, summarizing the key aspects of the content. Ensure that the Meta keywords and tags are distinct, relevant, and aligned with the content. Emphasize generating high-level, impactful keywords and tags for improved search engine visibility."
    },
    {
        "role": "user",
        "content": "Instructions:\n1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.\n2. Extract 5 to 8 high-volume Meta keywords from the given context, focusing on general, high-level terms.\n3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.\n4. If the Meta description exceeds 20 words, trim it accordingly.\n5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.\n6. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.\n7. Identify key tags central to the themes of security updates and Linux server management. Tags should have low difficulty and high search volume.\n8. Ensure each tag is unique and relevant to the content provided.\n9. List these tags in a format within brackets and separated by commas (e.g., [Tag1, Tag2, Tag3]).\n10. Reinforce the importance of uniqueness and relevance for the tags.\n11. Meta keywords and tags must be unique from each other\n\n.Given Context: {context}\n\nExample Output:\n{\n    \"Meta keywords\": [\"Keyword1\", \"Keyword2\", \"Keyword3\"],\n    \"Meta description\": \"A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.\",\n    \"Tags\": [\"Tag1\", \"Tag2\", \"Tag3\"]\n}\n\nJSON Output:"
    
    
    }
    ],


  10:  [
        {
            "role": "system",
            "content": (
                "You are an SEO specialist tasked with analyzing the provided context "
                "to identify Meta keywords, craft a concise Meta description, and generate "
                "tags optimized for SEO. The Meta description should be up to 20 words long, "
                "summarizing the key aspects of the content. Ensure that the Meta keywords and "
                "tags are distinct, relevant, and aligned with the content. Focus on generating "
                "high-level, impactful keywords and tags that are commonly searched within the "
                "industry for improved search engine visibility."
            )
        },
        {
            "role": "user",
            "content": (
                f"Instructions:\n"
                "1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.\n"
                "2. Extract 5 to 8 general, high-level Meta keywords from the given context, focusing on terms likely to have high search volumes.\n"
                "3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.\n"
                "4. If the Meta description exceeds 20 words, trim it accordingly.\n"
                "5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.\n"
                "6. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.\n"
                "7. Identify key tags central to the themes of security updates and Linux server management. Tags should be impactful and relevant.\n"
                "8. Ensure each tag is unique and relevant to the content provided.\n"
                "9. List these tags in a format within brackets and separated by commas (e.g., [Tag1, Tag2, Tag3]).\n"
                "10. Reinforce the importance of uniqueness and relevance for the tags.\n"
                "11. Meta keywords and tags must be unique from each other.\n\n"
                f"Given Context: {context}\n\n"
                "Example Output:\n"
                "{\n"
                "    \"Meta keywords\": [\"Keyword1\", \"Keyword2\", \"Keyword3\"],\n"
                "    \"Meta description\": \"A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.\",\n"
                "    \"Tags\": [\"Tag1\", \"Tag2\", \"Tag3\"]\n"
                "}\n\n"
                "JSON Output:"
            )
        }
    ],

11 : [
        {
            "role": "system",
            "content": (
                "You are an SEO specialist with expertise in identifying high-impact keywords and crafting optimized metadata for content. "
                "Analyze the provided text to extract relevant Meta keywords, create a concise Meta description, and generate tags. "
                "Focus on terms and phrases that are likely to have high search volumes and low competition based on industry knowledge. "
                "The Meta description should summarize the content in up to 20 words. Ensure the keywords and tags are relevant, impactful, and align with SEO best practices."
            )
        },
        {
            "role": "user",
            "content": (
                f"Instructions:\n"
                "1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.\n"
                "2. Extract 5 to 8 general, high-impact Meta keywords from the given context, focusing on terms likely to have high search volumes and low competition.\n"
                "3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.\n"
                "4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.\n"
                "5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.\n"
                "6. Identify key tags central to the themes of security updates and Linux server management. Tags should be impactful and relevant.\n"
                "7. Ensure each tag is unique and relevant to the content provided.\n"
                "8. List these tags in a format within brackets and separated by commas (e.g., [\"Tag1\", \"Tag2\", \"Tag3\"]).\n"
                "9. Emphasize the importance of uniqueness and relevance for the tags.\n"
                "10. Meta keywords and tags must be unique from each other.\n\n"
                f"Given Context: {context}\n\n"
                "Example Output:\n"
                "{\n"
                "    \"Meta keywords\": [\"Keyword1\", \"Keyword2\", \"Keyword3\"],\n"
                "    \"Meta description\": \"A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.\",\n"
                "    \"Tags\": [\"Tag1\", \"Tag2\", \"Tag3\"]\n"
                "}\n\n"
                "JSON Output:"
            )
        }
    ],

12 : [
        {
            "role": "system",
            "content": (
                "You are an SEO specialist with expertise in identifying high-impact keywords and crafting optimized metadata for content. "
                "Analyze the provided text to extract relevant Meta keywords, create a concise Meta description, and generate tags. "
                "Focus on terms and phrases that are likely to have high search volumes and low competition based on industry knowledge. "
                "The Meta description should summarize the content in up to 20 words. Ensure the keywords and tags are relevant, impactful, and align with SEO best practices."
            )
        },
        {
            "role": "user",
            "content": (
                f"Instructions:\n"
                "1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.\n"
                "2. Extract 5 to 8 general, high-impact Meta keywords from the given context, focusing on terms likely to have high search volumes and low competition.\n"
                "3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.\n"
                "4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.\n"
                "5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.\n"
                "6. Identify key tags central to the themes of Linux security, web application security, and SOC 2 compliance. Tags should be specific, targeted, and relevant for a platform like Joomla.\n"
                "7. Ensure each tag is unique and relevant to the content provided.\n"
                "8. List these tags in a format within brackets and separated by commas (e.g., [\"Tag1\", \"Tag2\", \"Tag3\"]).\n"
                "9. Emphasize the importance of uniqueness and relevance for the tags.\n"
                "10. Meta keywords and tags must be unique from each other the tags should be  5 .\n\n"
    
                f"Given Context: {context}\n\n" 
                "Example Output:\n"
                "{\n"
                "    \"Meta keywords\": [\"Keyword1\", \"Keyword2\", \"Keyword3\"],\n"
                "    \"Meta description\": \"A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.\",\n"
                "    \"Tags\": [\"Tag1\", \"Tag2\", \"Tag3\"]\n"
                "}\n\n"
                "JSON Output:"
            )
        }
    ],

13:[
        {
            "role": "system",
            "content": (
                "You are an SEO specialist with expertise in identifying high-impact keywords and crafting optimized metadata for content. "
                "Analyze the provided text to extract relevant Meta keywords, create a concise Meta description, and generate tags. "
                "Focus on terms and phrases that are likely to have high search volumes and low competition based on industry knowledge. "
                "The Meta description should summarize the content in up to 20 words. Ensure the keywords and tags are relevant, impactful, and align with SEO best practices."
            )
        },
        {
            "role": "user",
            "content": (
                f"Instructions:\n"
                "1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.\n"
                "2. Extract 5 to 8 general, high-impact Meta keywords from the given context, focusing on terms likely to have high search volumes and low competition.\n"
                "3. Create a concise Meta description that summarizes the given context, ensuring it does not exceed 20 words.\n"
                "4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.\n"
                "5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.\n"
                "6. Identify key tags central to the themes of Linux security, web application security, and SOC 2 compliance. Tags should be specific, targeted, and relevant for a platform like Joomla.\n"
                "7. Ensure each tag is unique and relevant to the content provided.\n"
                "8. List these tags in a format within brackets and separated by commas (e.g., [\"Tag1\", \"Tag2\", \"Tag3\"]).\n"
                "9. Emphasize the importance of uniqueness and relevance for the tags.\n"
                "10. Meta keywords and tags must be unique from each other.\n\n"
                f"Given Context: {context}\n\n"
                "Example Output:\n"
                "{\n"
                "    \"Meta keywords\": [\"Keyword1\", \"Keyword2\", \"Keyword3\"],\n"
                "    \"Meta description\": \"A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.\",\n"
                "    \"Tags\": [\"Tag1\", \"Tag2\", \"Tag3\"]\n"
                "}\n\n"
                "JSON Output:"
            )
        }
    ],

14 : [{"role": "system",
    "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites."
},
{
    "role": "user",
    "content": f"""Instructions:
    1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
    2. Extract 5 to 10 high-volume Meta keywords from the given context.
    3. Create a concise Meta description that summarizes the given context and ensure that the Meta description Words length should not exceed 20.
    4. If the Meta description Words length is greater than 20 then trim the Meta description that is more than 20 Words.
    4. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
    5. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.

    6. Ensure that each Meta keyword is distinct from the others. The provided tags above are preferred, even if they consist of multiple words.
    7. Clarify the objective: "Extract Meta tags that reflect key concepts and terms from the text, focusing on security and Linux server management." This provides clear direction for what the tags should represent.

    8. Ensure these tags are direct, with each tag being a single impactful word or multiple impactful words.

    9. Specify the format for listing tags: "List these tags in a format within brackets and separated by commas." This ensures consistency in how the tags are presented, like this: [Tag1, Tag2, Tag3].

    10. Reinforce the importance of uniqueness and relevance: "Tags should be unique, relevant, and derived strictly from the content provided." This emphasizes the quality criteria for the tags.
    11. Meta keywords and Tags must be unique from each other or similar but not exactly the same.

    Given Context: {context}

    Example Output:{{
    "Meta Keywords": ["Linux web apps", "SOC 2 compliance", "web app vulnerabilities", "risk assessment", "penetration testing"],
    "Meta Description": "Secure Linux web apps for SOC 2 compliance and data security.",
    "Meta Tags": [Linux security, compliance standards, web app vulnerabilities, SOC 2, risk assessment, penetration testing]
    }}
    """}
    ],

15: [
    {"role": "system",
    "content": "As an SEO specialist, analyze the provided context to identify Meta keywords and craft a concise Meta description for SEO optimization. The Meta description should be up to 20 words long. Ensure that the Meta keywords are distinct, surrounded by quotes, and directly relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Highlight the significance of adhering to SEO best practices for effective website optimization."
},
{
    "role": "user",
    "content": f"""Given the context below, extract Meta keywords and create a Meta description that encapsulates the essence of the content. Ensure that the Meta keywords are high-impact terms relevant to the specific topic discussed. Format each Meta keyword in quotes and separate them with commas. Additionally, list Meta tags in brackets, separated by commas.

    Context:
    {context}

    Specific Instructions:
    1. Identify Meta keywords that represent key concepts and terms related to the specific topic discussed in the context.
    2. Craft a Meta description that summarizes the content within 20 words, emphasizing its significance.
    3. Format Meta keywords in quotes and provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description.
    4. List Meta tags in brackets, ensuring they are relevant terms derived from the content.

    Example Output:
    {{
        "Meta Keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
        "Meta Description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 20 words.",
        "Meta Tags": [Tag1, Tag2, Tag3, Tag4, Tag5]
    }}
    """}
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

    prompt_key =  int(get_prompt_key) if get_prompt_key  else 1 
    try:
        print("Prompt key used  : ", prompt_key)
        prompt = promts_dict(prompt_key,context)
        return prompt
    except Exception as e:
        print(e)
