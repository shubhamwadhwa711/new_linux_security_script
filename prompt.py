import configparser
import os
import json

config = configparser.ConfigParser(interpolation=None)
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))
title = config.get("chatgpt", "title")


def promts_dict(key, context):
    prompts = {
        1: f""" Instructions:
            1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
            2. Extract 5 to 10 high-volume Meta keywords and meta tags from the given context.
            3. Create a concise Meta description that summarizes the given context, ensuring that the Meta description's word length does not exceed 20 words.
            4. If the Meta description's word length exceeds 20 words, trim it to meet the limit.
            5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
            6. The Meta keywords, meta tags, and Meta description must be strictly derived from the provided context, aligning with SEO best practices.
            7. Ensure that each Meta keyword is distinct from the others.
            8. Extract Meta tags that reflect key concepts and terms from the text, focused on infosecurity and Linux server management.
            9. Ensure these tags are specific, with each tag being a single impactful word or multiple authoritative words.
            10. List these tags in a format within brackets and separated by commas.
            11. Tags should be unique, relevant, and derived strictly from the content provided.
            12. Meta keywords and Tags must be unique from each other or similar but not exactly the same.
            13. Avoid words common to all content related to Linux and open source, including "linux" and "open source" and "cybersecurity".
            14. Results must be suitable for a Linux and info security professional audience who are generally highly-educated and experienced Linux sysadmins.
            15. The collection of meta tags should strictly relate to the context.
            16. Strictly adhere to SEO best practices in generating accurate Meta data for websites.
            17. Tags should be chosen for their low keyword difficulty and high search volume, relevant for boosting search engine visibility.
            18. Create primary single-word tags that are central to the themes of Linux, open source, infosecurity, and Linux server management.
            19. Keywords and tags must not be the same.
            20 . Strictly follow these steps:
                    1. Read all the previous process context.
                    2 .Check if the tags are similar to the last data (all processed data).
                    3 . Ensure the tags are general and valid. For example, use "Security Advisory" instead of a specific term like "Red Hat Security Advisory." Always apply this rule
                    4.Maintain the state of the conversation, including user preferences, prior questions, and given answers.
                


            - Keywords: high-level words or phrases that capture the primary topics and themes of the content.
            - Tags: low-level general labels that categorize the content into specific themes. These help with site navigation and categorizing content. Tags provide details that the keywords don’t convey, so it’s handy to use them together. 

            After creating tags and keywords for this context, review above instructions again to ensure you have not used words like "linux" and "open source" and "cybersecurity" in your results.

            ###
            Given Context: {context}"""
        # [
        # {
        #             "role": "system",
        #             "content": "You are an SEO expert. Analyze the given context from a website for Linux and open-source security professionals to create meta tags, meta keywords, and a concise meta description for SEO purposes. Provide structured JSON output with the top 5 to 8 Meta keywords, meta tags, and the Meta description. The Meta description should be up to 20 words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context."
        # },
        #                 {
        #                     "role": "user",
        #                     "content": f"""
        #     Instructions:
        #     1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
        #     2. Extract 5 to 10 high-volume Meta keywords and meta tags from the given context.
        #     3. Create a concise Meta description that summarizes the given context, ensuring that the Meta description's word length does not exceed 20 words.
        #     4. If the Meta description's word length exceeds 20 words, trim it to meet the limit.
        #     5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
        #     6. The Meta keywords, meta tags, and Meta description must be strictly derived from the provided context, aligning with SEO best practices.
        #     7. Ensure that each Meta keyword is distinct from the others.
        #     8. Extract Meta tags that reflect key concepts and terms from the text, focused on infosecurity and Linux server management.
        #     9. Ensure these tags are specific, with each tag being a single impactful word or multiple authoritative words.
        #     10. List these tags in a format within brackets and separated by commas.
        #     11. Tags should be unique, relevant, and derived strictly from the content provided.
        #     12. Meta keywords and Tags must be unique from each other or similar but not exactly the same.
        #     13. Avoid words common to all content related to Linux and open source, including "linux" and "open source" and "cybersecurity".
        #     14. Results must be suitable for a Linux and info security professional audience who are generally highly-educated and experienced Linux sysadmins.
        #     15. The collection of meta tags should strictly relate to the context.
        #     16. Strictly adhere to SEO best practices in generating accurate Meta data for websites.
        #     17. Tags should be chosen for their low keyword difficulty and high search volume, relevant for boosting search engine visibility.
        #     18. Create primary single-word tags that are central to the themes of Linux, open source, infosecurity, and Linux server management.
        #     19. Keywords and tags must not be the same.
        #     Difference between "keywords" and "tags":
        #     - Keywords: high-level words or phrases that capture the primary topics and themes of the content.
        #     - Tags: low-level general labels that categorize the content into specific themes. These help with site navigation and categorizing content. Tags provide details that the keywords don’t convey, so it’s handy to use them together.
        #     After creating tags and keywords for this context, review above instructions again to ensure you have not used words like "linux" and "open source" and "cybersecurity" in your results.
        #     ###
        #     Given Context: {context}
        #     ###
        #     Example Output:
        #     {{
        #         "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
        #         "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
        #         "Tags": ["Tag1", "Tag2", "Tag3"]
        #     }}
        #     ###
        #     JSON Output:
        #     """
        #         }
        #     ],
        ,
        2: [
            {
                "role": "system",
                "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites.                     Analyze the provided text related to text. Identify key single-word tags that are central to the themes of security updates and Linux server management.These tags should be chosen for their low difficulty and high search volume, relevant for boosting search engine visibility ",
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


    JSON Output:""",
            },
        ],
        3: [
            {
                "role": "system",
                "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 Words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites.                     Analyze the provided text related to text. Identify key single-word tags that are central to the themes of security updates and Linux server management.These tags should be chosen for their low difficulty and high search volume, relevant for boosting search engine visibility ",
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


    JSON Output:""",
            },
        ],
        4: [
            {
                "role": "system",
                "content": "Act as an SEO specialist and analyze the given context to identify Meta keywords and create a concise Meta description for SEO purposes. The Meta description should be up to 20 words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context. Provide a structured JSON output with the top 5 to 8 Meta keywords and the Meta description. Emphasize the importance of SEO best practices in generating accurate Meta data for websites. Analyze the provided text related to the given context. Identify key single-word tags that are central to the themes of security updates and Linux server management. These tags should be chosen for their low difficulty and high search volume, relevant for boosting search engine visibility.",
            },
            {
                "role": "user",
                "content": f"""Instructions:
        1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
        2. Extract 5 to 10 high-volume Meta keywords from the given context.
        3. Create a concise Meta description that summarizes the given context, ensuring that the Meta description's word length does not exceed 20 words.
        4. If the Meta description's word length exceeds 20 words, trim it to meet the limit.
        5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
        6. The Meta keywords and Meta description must be directly derived from the provided context, aligning with SEO best practices.
        7. Ensure that each Meta keyword is distinct from the others. The provided tags above are preferred, even if they consist of multiple words.
        8. Clarify the objective: "Extract Meta tags that reflect key concepts and terms from the text, focusing on security and Linux server management." This provides clear direction for what the tags should represent.
        9. Ensure these tags are direct, with each tag being a single impactful word or multiple impactful words and also the length of tags  is 5.
        10. Specify the format for listing tags: "List these tags in a format within brackets and separated by commas." This ensures consistency in how the tags are presented, like this: [Tag1, Tag2, Tag3].
        11. Reinforce the importance of uniqueness and relevance: "Tags should be unique, relevant, and derived strictly from the content provided." This emphasizes the quality criteria for the tags.
        12. Meta keywords and Tags must be unique from each other or similar but not exactly the same.
        13. Avoid using common words such as "Linux," "open source," and "security."
 
        Difference between "keywords" and "tags":
        - Keywords: Specific words or phrases that capture the primary topics and themes of the content. Used in Meta tags for SEO to improve search engine visibility.
        - Tags: General labels that categorize the content into broader themes. Help with site navigation and organizing content.
 
        ###
        Given Context: {context}
 
        ###
        Example Output:
        {{
            "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
            "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
            "Tags": ["Tag1", "Tag2", "Tag3"]
        }}
        ###
        JSON Output:
        """,
            },
        ],
        5: [
            # {
            #             "role": "system",
            #             "content": "You are an SEO expert. Analyze the given context from a website for Linux and open-source security professionals to create meta tags, meta keywords, and a concise meta description for SEO purposes. Provide structured JSON output with the top 5 to 8 Meta keywords, meta tags, and the Meta description. The Meta description should be up to 20 words long. Ensure the Meta keywords are distinct, surrounded by quotes, and relevant to the given context."
            # },
            {
                "role": "user",
                "content": f"""
        Instructions:
        1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
        2. Extract 5 to 10 high-volume Meta keywords and meta tags from the given context.
        3. Create a concise Meta description that summarizes the given context, ensuring that the Meta description's word length does not exceed 20 words.
        4. If the Meta description's word length exceeds 20 words, trim it to meet the limit.
        5. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
        6. The Meta keywords, meta tags, and Meta description must be strictly derived from the provided context, aligning with SEO best practices.
        7. Ensure that each Meta keyword is distinct from the others.
        8. Extract Meta tags that reflect key concepts and terms from the text, focused on infosecurity and Linux server management.
        9. Ensure these tags are specific, with each tag being a single impactful word or multiple authoritative words.
        10. List these tags in a format within brackets and separated by commas.
        11. Tags should be unique, relevant, and derived strictly from the content provided.
        12. Meta keywords and Tags must be unique from each other or similar but not exactly the same.
        13. Avoid words common to all content related to Linux and open source, including "linux" and "open source" and "cybersecurity".
        14. Results must be suitable for a Linux and info security professional audience who are generally highly-educated and experienced Linux sysadmins.
        15. The collection of meta tags should strictly relate to the context.
        16. Strictly adhere to SEO best practices in generating accurate Meta data for websites.
        17. Tags should be chosen for their low keyword difficulty and high search volume, relevant for boosting search engine visibility.
        18. Create primary single-word tags that are central to the themes of Linux, open source, infosecurity, and Linux server management.
        19. Keywords and tags must not be the same.

        Difference between "keywords" and "tags":
        - Keywords: high-level words or phrases that capture the primary topics and themes of the content.
        - Tags: low-level general labels that categorize the content into specific themes. These help with site navigation and categorizing content. Tags provide details that the keywords don’t convey, so it’s handy to use them together. 

        After creating tags and keywords for this context, review above instructions again to ensure you have not used words like "linux" and "open source" and "cybersecurity" in your results.

        ###
        Given Context: {context}
        ###
        Example Output:
        {{
            "Meta keywords": ["Keyword1", "Keyword2", "Keyword3"],
            "Meta description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
            "Tags": ["Tag1", "Tag2", "Tag3"]
        }}
        ###
        JSON Output:
        """,
            }
        ],
        6: [
            {
                "role": "user",
                "content": f""" 
Instructions for the Assistant:
The requirement is to create keywords, meta tags, meta description, title tag and H1 tag for an article related to Linux and open source security for a Linux admin and infosec professional audience. Results should be suitable for a sophisticated, professional audience specialized in Linux system administration and cybersecurity.

Keyword requirements:
Extract 3-5 meta keywords relevant to the context without using overly generic terms relevant to the article's context, focusing on high-volume terms

Tags requirements:
Create a list of 4-6 relevant high-volume tags based on the specified criteria and context provided without using overly generic terms like "Linux" and "cybersecurity", focusing instead on the specific topics of interest directly related to the content. Tags should be case-insensitive, singular form, and in natural English without underscores or hyphens. Exclude specific tags like SLE-15-SP4, CVE-2023-38408, CVE-2023-3446, libopenssl1_1-1.1.1l-150400.7.48.1, sles15-image-15.0.0-27.14.85, SUSE-SU-2024:1437-1 and any others that are not natural English words.

Meta description requirements:
The created meta description should not exceed 20 words.  Compose a concise meta description summarizing the article's content within a 20-word limit

Title Tag Requirements:
Generate a title tag according to SEO best practices. Title tags and H1 tags must not be the same. Title can be no longer than seven words or 60 characters. Title tag must be unique from any other Title tag. For context of type "Advisory", titles must have non-English words like SLE-15-SP4 or CVE-2023-38408 or CVE-2023-3446 or ELSA-2024-3954, taken from content in the article to distinguish it from any other title tag. Include the level of detail necessary to ensure no other Title Tag has the same content. Include the application name in the Title Tag where it exists, like gdk-pixbuf2-2.42.6-4 or other data involving unique numbers or letters.

H1 Tag Requirements:
Generate an H1 tag according to SEO best practices. Title and H1 tags must not be the same. H1 Tag can be no longer than ten words or 100 characters. H1 Tag should be very descriptive and contain more detail about the article than title tags. H1 tags should be closer to a description of the main synopsis of the article. Include one of the unique keywords in the title to make the title unique.

Overall requirements:
Results must be suitable for a Linux and info security professional audience who are generally highly-educated and experienced Linux sysadmins.

Identify the type of article as one of Advisory, which is content related to a security update of some kind, or Feature, which is a piece of news or a story that digs deep on a particular topic. 

After creating tags and keywords and titles and H1 tags for this context, review above instructions again to ensure you have not used words like "linux" and "open source" and "open source security" and "cybersecurity" or other terms or phrases that generically apply to any content related to Linux and open source security in your results.

###
Given Context: {context}
###
Example Output:
{{
    "Keywords": ["Keyword1", "Keyword2", "Keyword3"],
    "Description": "A succinct summary that encapsulates the main points of the content, optimized for search engines and not exceeding 160 characters.",
    "Tags": ["Tag1", "Tag2", "Tag3", "Tag4"],
    "Title": "Linux Security Article Title",
    "H1": "Linux Security H1 Tag",
    "Type": "Feature or Advisory"
}}
###
JSON Output:
"""
            }
        ],
    }

    return prompts[key]


gpt_model = {
    1: "gpt-3.5-turbo-1106",
    2: "gpt-4",
    3: "gpt-3.5-turbo"
}


def get_model():
    get_gpt_model = config.get("chatgpt", "gpt_model")
    model = int(get_gpt_model[0]) if get_gpt_model else 1
    try:
        return gpt_model.get(model)
    except Exception as e:
        print(e)


def get_prompt(context):
    get_prompt_key = config.get("chatgpt", "prompt_key")

    prompt_key = int(get_prompt_key) if get_prompt_key else 1
    try:
        # dave print("Prompt key used  : ", prompt_key)
        prompt = promts_dict(prompt_key, context)
        return prompt
    except Exception as e:
        print(e)
