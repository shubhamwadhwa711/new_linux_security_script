import configparser
import os
import json

config = configparser.ConfigParser(interpolation=None)
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))


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
Instructions:
1. Analyze the provided text to identify the most relevant Meta keywords for SEO optimization.
2. Extract 3 to 5 high-volume Meta keywords from the given context.
3. Extract 3 to 5 tags from the given context.
4. Create a concise Meta description that summarizes the given context, ensuring that the Meta description's word length does not exceed 20 words.
5. Use keywords that precisely describe the content.
6. Ensure keywords are directly related to the main topics and themes of the content.
7. Exclude terms that are too broad or unrelated to the core content.
8. Use descriptive phrases that capture the essence of the content.
9. For example, "Security update for OpenSSH" is more descriptive and useful than just "Security update."
10. Please return only the tags that do not contain "_" or "-".
11. Treat the tags as case-insensitive, meaning "System Update" and "System update" are considered the same tag.
12. Exclude specific tags like "SLE-15-SP4, CVE-2023-38408, CVE-2023-3446, libopenssl1_1-1.1.1l-150400.7.48.1, sles15-image-15.0.0-27.14.85, SUSE-SU-2024:1437-1".
13. keep ony one common tag for example either "Update" or "System Update". either use common phrase applicable to all.
14. Keep singular form of verb rather than plural for example "CVE" and "CVEs", keep only singular form i.e "CVE".
15. Keep keywords concise and to the point.
16. Avoid adding filler words that don't contribute meaningful information.
17. Consider what users might search for when looking for content similar to yours.
18. Include keywords that match user queries and search intent.
19. Avoid overly generic terms like "Summary" or "Released" or "Updates", "Ubuntu" that don't provide specific information about the content.
20. Meta description must contains few generated keywords as well to help in SEO.
21. If the Meta description's word length exceeds 20 words, trim it to meet the limit.
22. Format each Meta keyword in quotes and separate them with commas, ensuring no repetition and high relevance to the context.
23. The Meta keywords, tags, and Meta description must be strictly derived from the provided context, aligning with SEO best practices.
24. Ensure that each Meta keyword is distinct from the others.
25. Extract tags that reflect key concepts and terms from the text, focused on infosecurity and Linux server management.
26. Ensure these tags are broad and general, with each tag being a single impactful word.
27. List these tags in a format within brackets and separated by commas.
28. Tags should be unique, relevant, and derived strictly from the content provided.
29. Meta keywords and Tags must be unique from each other.
30. Avoid words common to all content related to Linux and open source, including "linux" and "open source" and "cybersecurity".
31. Results must be suitable for a Linux and info security professional audience who are generally highly-educated and experienced Linux sysadmins.
32. The collection of tags should strictly relate to the context.
33. Strictly adhere to SEO best practices in generating accurate Meta data for websites.
34. Tags should be chosen for their relevance to the broader themes and topics of the content.
35. Create primary single-word tags that are central to the themes of Linux, open source, infosecurity, and Linux server management.

Difference between "keywords" and "tags":
- Keywords: high-level words or phrases that capture the primary topics and themes of the content.
- Tags: broad, general labels that categorize the content into specific themes. These help with site navigation and categorizing content. Tags provide details that the keywords don’t convey, so it’s handy to use them together.

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
        print("Prompt key used  : ", prompt_key)
        prompt = promts_dict(prompt_key, context)
        return prompt
    except Exception as e:
        print(e)
