context = """Purpose: The primary role of this agent is to assist users by analyzing code. It should
            be able to generate code and answer questions about code provided. """

code_parser_template = """Parse the response from a previous LLM into a description and a string of valid code, 
                            also come up with a valid filename this could be saved as that doesnt contain special characters 
                            like triple or double quotes in the beginning or ending of the string, or keywords like python. 
                            Here is the response: {response}. You should parse this in the following JSON Format: """