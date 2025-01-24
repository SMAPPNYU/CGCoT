import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
from joblib import delayed, Parallel
import copy

"""
To prevent overusing the OpenAI API (or whatever API you choose to use), this class defines several methods that will be used sequentially.
The order of operations is as follows:
1. Instantiate a CGCoT_Breakdowns object with a dataframe of text
2. Use `parallel_get_breakdowns` in CGCoT_Breakdowns to get breakdowns for all texts
3. Use `build_csb_text` in CGCoT_Breakdowns to get a final concept-specific breakdown, formatted as a string, for all texts
4. Use a sampling method of your choice to pair together texts
5. Instantiate a CGCoT_PairwiseComparisons object with a dataframe of the texts to pairwise compare, the pairwise comparison prompt, and 
   the prompt used to extract the final answer from each pairwise comparison
6. Generate the final prompts for each pairwise comparison using `generate_pairwise_comparison_prompts`
7. Make pairwise comparisons using `parallelize_openai_calls` with the pairwise comparison prompts
8. Generate the final extraction prompts for each pairwise comparison using `generate_extraction_prompts`
9. Extract the final answers using `parallelize_openai_calls` with the extraction prompts
10. Create the final pairwise comparison results dataframe using `create_win_df`
11. This final dataframe of wins and losses can be used to calculate scores for each text using the Bradley-Terry model
"""

class CGCoT_Breakdowns:
    """
    A Concept-Guided Chain-of-Thought (CGCoT) text scoring framework leveraging large language models (LLMs).
    Described in detail here: https://ieeexplore.ieee.org/document/10825235.
    This class provides methods to generate the concept-specific breakdowns.
    
    Attributes:
        df (pd.DataFrame): Input DataFrame containing text and text_id variables.
        text_var (str): Column name for the text variable in `df`.
        text_id_var (str): Column name for the ID variable for each text in `df`.
        conceptguided_questions (list): List of researcher-designed prompts for concept-specific breakdowns.
        text (list): List of text from the `text_var` column in `df`.
        text_id (list): List of IDs from the `text_id_var` column in `df`.
        id_text_dict (dict): Mapping of IDs to text from `text_var`.
        client (OpenAI): Initialized OpenAI client for LLM interactions.
    """
    
    def __init__(self,
                 df,
                 text_var,
                 text_id_var,
                 conceptguided_questions):
        
        self.df = df
        self.conceptguided_questions = conceptguided_questions
        self.text_id_var = text_id_var
        self.text_var = text_var
        
        self.text = list(df[self.text_var])
        self.text_id = list(df[self.text_id_var])
        
        self.id_text_dict = {n:t for n,t in zip(self.text_id, self.text)}
        
        self.client = OpenAI()
        
    def conceptguided_prompting_openai(self,
                                       text,
                                       system_prompt,
                                       model='gpt-4o-mini',
                                       temp=1.0):
        """ 
        Generates a concept-specific breakdown (CSB) for a given text using concept-guided questions and OpenAI's API.

        This method iteratively applies a series of concept-guided questions to the input `text`, using an 
        OpenAI language model to provide responses. It builds a chain of interactions based on the system prompt and 
        user-defined questions, capturing responses as a breakdown of the concept of interest.
        
        Parameters:
        text (str): The input text to be analyzed using the concept-guided questions.
        system_prompt (str): The system-level prompt that provides the overall context for the OpenAI model.
        model (str, optional): The OpenAI model to use for generating responses. Defaults to 'gpt-4o-mini'.
        temp (float, optional): The temperature parameter controlling randomness in the model's responses. 
                                Defaults to 1.0.
        Returns:
            csb: A list of strings where each string represents the OpenAI model's response to one concept-guided question (csb - Concept-Specific Breakdown).

            Raises:
                Exception: If the OpenAI client encounters an issue during a request, retries are attempted using 
                           exponential backoff. Errors are printed, and the method sleeps before retrying.

            Side Effects:
                - Prints retry messages (`uh oh spaghetti-os <time>`) if API calls fail and retries are necessary.

            Notes:
                - The method utilizes a `sleepy_times` list for exponential backoff in case of repeated failures 
                  during API calls.
                - The `self.conceptguided_questions` list is modified to include the input `text` (the original text) in the first question.
        """

        sleepy_times = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]

        messages = [{"role": "system", "content": system_prompt}]

        # this holds all the aspects of the concept-specific breakdown (CSB)
        csb = []

        # Load the text into the first concept-guided question
        loaded_cgqs = copy.copy(self.conceptguided_questions)
        loaded_cgqs[0] = 'Text: "' + text + '"\n\n' + self.conceptguided_questions[0]

        for j in range(len(loaded_cgqs)):
            messages.append({"role": "user", "content": loaded_cgqs[j]})

            for i in range(len(sleepy_times)):
                try:
                    completion = self.client.chat.completions.create(model=model,
                                                                     messages=messages,
                                                                     temperature=temp)
                    response = completion.choices[0].message.content
                    break
                except:
                    print('uh oh spaghetti-os ' + str(sleepy_times[i]))
                    time.sleep(sleepy_times[i])

            messages.append({"role": "assistant", "content": response})
            csb.append(response)

        return csb

    def parallel_get_breakdowns(self,
                                chunk_size,
                                system_prompt,
                                model='gpt-4o-mini',
                                temp=1.0):
                                    
        """
        Parallelizes the generation of concept-specific breakdowns (CSBs) for a dataset using OpenAI's API.

        This method splits the input texts into chunks and uses parallel processing to generate CSBs efficiently. 
        It leverages the `conceptguided_prompting_openai` method for individual text analysis and aggregates the 
        results into a dictionary mapping IDs to their respective breakdowns.

        Parameters:
            chunk_size (int): The number of texts to process in parallel in each chunk.
            system_prompt (str): The system-level prompt that provides overall context for the OpenAI model.
            model (str, optional): The OpenAI model to use for generating responses. Defaults to 'gpt-4o-mini'.
            temp (float, optional): The temperature parameter controlling randomness in the model's responses. 
                                    Defaults to 1.0.

        Returns:
            id_gsb_dict: A dictionary mapping IDs to their respective concept-specific breakdowns (CSBs), where each 
                         breakdown is a list of responses from the OpenAI model.

        Side Effects:
            - Displays a progress bar (using `tqdm`) to indicate the processing status of text chunks.

        Notes:
            - The method uses the `chunked_data` helper function to divide the text data into smaller chunks of size `chunk_size`.
            - Parallel processing is implemented using `joblib.Parallel` with a threading backend to optimize API requests.
        """

        # collect the concept-specific breakdowns
        csb_collection = []

        # get the text data
        text_data = []
        
        for i in self.text_id:
            text_data.append(self.id_text_dict[i])

        for text_data_chunk in tqdm(self.chunked_data(text_data, chunk_size), total=len(text_data)//chunk_size):
            results = Parallel(n_jobs=chunk_size, backend='threading')(delayed(self.conceptguided_prompting_openai)(t, system_prompt, model, temp) for t in text_data_chunk)
            csb_collection.extend(results)

        id_gsb_dict = {n:t for n,t in zip(self.text_id, csb_collection)}

        return id_gsb_dict

    def chunked_data(self, text_data, chunk_size):
        """
        Helper method for parallelizing API requests by successively chunking given text data
        """
        for i in range(0, len(text_data), chunk_size):
            yield text_data[i:i+chunk_size]
    
    def build_csb_text(self, og_id_breakdown_dict, breakdown_headers=None):
        """
        Constructs a formatted text representation of concept-specific breakdowns (CSBs) for each input text.

        This method combines the original text with its corresponding breakdown, optionally including headers 
        for each breakdown component. The result is a dictionary where each key is an ID and the value is a 
        formatted text summary of the breakdown.

        Parameters:
            og_id_breakdown_dict (dict): A dictionary mapping IDs to their concept-specific breakdowns, 
                                         where each breakdown is a list of strings.
            breakdown_headers (list, optional): A list of headers corresponding to the elements of the breakdown. 
                                                If provided, headers are prepended to each breakdown component. 
                                                Defaults to `None`.

        Returns:
            id_breakdown_dict (dict): A dictionary mapping IDs to formatted strings, where each string includes the original text 
                                      followed by its breakdown.

        Notes:
            - If `breakdown_headers` is provided, each breakdown component will be prefixed with its corresponding 
              header (e.g., "Header: Component").
            - If `breakdown_headers` is not provided, only the breakdown components are included in the text.
        """
        id_breakdown_dict = {}

        # build the breakdowns
        for i in self.text_id:
            summary_per = 'Original Text: "' + self.id_text_dict[i] + '"\n\nBreakdown of Text:'
            if breakdown_headers is not None:
                for j in range(len(og_id_breakdown_dict[i])):
                    summary_per = summary_per + '\n' + breakdown_headers[j] + ': ' + og_id_breakdown_dict[i][j]
            else:
                for j in range(len(og_id_breakdown_dict[i])):
                    summary_per = summary_per + '\n' + str(j+1) + ') ' + og_id_breakdown_dict[i][j]
            id_breakdown_dict[i] = summary_per

        return id_breakdown_dict
    

class CGCoT_PairwiseComparisons:
    """
    A Concept-Guided Chain-of-Thought (CGCoT) text scoring framework leveraging large language models (LLMs).
    Described in detail here: https://ieeexplore.ieee.org/document/10825235.
    This class provides methods to make pairwise comparisons between concept-specific breakdowns.

    Attributes:
        df (pd.DataFrame): Input DataFrame containing text paired up and ID variables.
        text_var1 (str): Column name for the first text variable in `df`.
        text_var2 (str): Column name for the second text variable in `df`.
        id_var (str): Column name for the ID variable for each pairwise comparison in `df`.
        pairwise_comparison_prompt (str): Prompt for pairwise text comparisons.
        extraction_prompt_text (str): Prompt for extracting the answer from pairwise comparisons.
        ids (list): List of unique IDs for pairwise comparison from the `id_var` column in `df`.
        text1 (list): List of text from the `text_var1` column in `df`.
        text2 (list): List of text from the `text_var2` column in `df`.
        id_text1_dict (dict): Mapping of IDs to text from `text_var1`.
        id_text2_dict (dict): Mapping of IDs to text from `text_var2`.
        client (OpenAI): Initialized OpenAI client for LLM interactions.
    """
    def __init__(self,
                 df,
                 text_var1,
                 text_var2,
                 breakdown_var1,
                 breakdown_var2,
                 id_var,
                 pairwise_comparison_prompt,
                 extraction_prompt_text):

        self.df = df
        self.text_var1 = text_var1
        self.text_var2 = text_var2
        self.breakdown_var1 = breakdown_var1
        self.breakdown_var2 = breakdown_var2
        self.id_var = id_var
        
        self.pairwise_comparison_prompt = pairwise_comparison_prompt
        self.extraction_prompt_text = extraction_prompt_text

        self.ids = list(self.df[id_var])
        self.text1 = list(self.df[text_var1])
        self.text2 = list(self.df[text_var2])
        self.breakdown1 = list(self.df[breakdown_var1])
        self.breakdown2 = list(self.df[breakdown_var2])

        self.id_text1_dict = {n:t for n,t in zip(self.ids, self.text1)}
        self.id_text2_dict = {n:t for n,t in zip(self.ids, self.text2)}
        
        self.id_breakdown1_dict = {n:t for n,t in zip(self.ids, self.breakdown1)}
        self.id_breakdown2_dict = {n:t for n,t in zip(self.ids, self.breakdown2)}

        # initiate the OpenAI client
        self.client = OpenAI()

    
    def chunked_data(self, text_data, chunk_size):
        """
        Helper method for parallelizing API requests by successively chunking given text data
        """
        for i in range(0, len(text_data), chunk_size):
            yield text_data[i:i+chunk_size]

    
    def generate_pairwise_comparison_prompts(self):
        """
        Generates pairwise comparison prompts based on concept-specific breakdowns for text pairs.

        This method constructs prompts by combining the concept-specific breakdowns of two 
        texts with a predefined pairwise comparison prompt template (`self.pairwise_comparison_prompt`). Each 
        prompt includes the breakdowns and a question or instruction for comparison.

        Returns:
            list: A list of pairwise comparison prompts, where each prompt contains the breakdowns of two texts 
                  and the pairwise comparison prompt
        """
        prompts = []

        for j in self.ids:
            p = 'Description 1: "' + self.id_breakdown1_dict[j] + '"\n\nDescription 2: "' + self.id_breakdown2_dict[j] +\
                '"\n\n' + self.pairwise_comparison_prompt
            prompts.append(p)

        return prompts

    def prompting_openai(self,
                         prompt,
                         system_prompt='You are a helpful assistant.',
                         model='gpt-4o-mini',
                         temp=1.0):
        """
        Sends a prompt to OpenAI's API and retrieves a response, with retries and exponential backoff on failure.
        
        This is used for making the pairwise comparisons and for extracting the final answer from each pairwise comparison.

        This method constructs a conversation context with a system prompt and user prompt, sends it to the specified 
        OpenAI model, and returns the model's response. If an error occurs during the API request, the method retries 
        using an exponential backoff strategy.

        Parameters:
        prompt (str): The main user prompt to be sent to the OpenAI model.
        system_prompt (str, optional): The system-level instruction that sets the context for the assistant. 
                                       Defaults to 'You are a helpful assistant.'.
        model (str, optional): The OpenAI model to use for generating responses. Defaults to 'gpt-4o-mini'.
        temp (float, optional): The temperature parameter controlling randomness in the model's responses. 
                                Defaults to 1.0.
        """

        sleepy_times = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
            ]

        for i in range(len(sleepy_times)):
            try:
                completion = self.client.chat.completions.create(model=model,
                                                                 messages=messages,
                                                                 temperature=temp)
                response = completion.choices[0].message.content
                break
            except Exception:
                print(Exception)
                print('uh oh spaghetti-os ' + str(sleepy_times[i]))
                time.sleep(sleepy_times[i])
        return response

    def parallelize_openai_calls(self,
                                 prompts_data,
                                 chunk_size,
                                 system_prompt='You are a helpful assistant.',
                                 model='gpt-4o-mini',
                                 temp=1.0):
        """
        Parallelizes OpenAI API calls for processing multiple prompts efficiently.

        This method divides the input data into smaller chunks and uses parallel processing to send multiple 
        prompts to the OpenAI API concurrently. Each prompt is processed using the `prompting_openai` method, 
        and the results are aggregated into a single collection.

        Parameters:
            prompts_data (list): A list of prompts to be sent to the OpenAI API.
            chunk_size (int): The number of prompts to process in parallel in each chunk.
            system_prompt (str, optional): The system-level instruction that sets the context for the assistant. 
                                           Defaults to 'You are a helpful assistant.'.
            model (str, optional): The OpenAI model to use for generating responses. Defaults to 'gpt-4o-mini'.
            temp (float, optional): The temperature parameter controlling randomness in the model's responses. 
                                    Defaults to 1.0.

        Returns:
            list: A list of responses from the OpenAI API, where each response corresponds to a prompt in 
                  `prompts_data`.

        Side Effects:
            - Displays a progress bar (using `tqdm`) to indicate the processing status of prompt chunks.

        Notes:
            - The method uses the `chunked_data` helper function to divide `prompts_data` into smaller chunks.
            - Parallel processing is implemented using `joblib.Parallel` with a threading backend for efficiency.
            - This method assumes that the `prompting_openai` method is already defined and functional.

        """
        results_collection = []
        for prompt_chunk in tqdm(self.chunked_data(prompts_data, chunk_size), total=len(prompts_data)//chunk_size):
            results = Parallel(n_jobs=chunk_size, backend='threading')(delayed(self.prompting_openai)(p, system_prompt, model, temp) for p in prompt_chunk)
            results_collection.extend(results)
        return results_collection

    def generate_extraction_prompts(self, pairwise_comparison_responses):
        """
        Generates a list of extraction prompts based on pairwise comparison responses.

        This method constructs prompts by combining each pairwise comparison response with a predefined 
        extraction prompt (`self.extraction_prompt_text`). These prompts are used to extract which text the LLM chose
        in pairwise comparisons.

        Parameters:
            pairwise_comparison_responses (list): A list of pairwise comparison responses, where each response 
                                                  is a string representing the input for prompt generation.

        Returns:
            extraction_prompts: A list of extraction prompts, where each prompt is a string that combines a pairwise comparison 
                                response with the extraction prompt.
        """
        extraction_prompts = []

        for j in pairwise_comparison_responses:
            sent = 'Text: "' + j + '"\n\n' + self.extraction_prompt_text
            extraction_prompts.append(sent)

        return extraction_prompts

    def check_extraction_text(self,
                              extraction_text,
                              pairwise_comparison_responses,
                              valid_responses):
        """
        Checks the validity of extraction outputs against a set of valid responses.

        This helper function iterates through the `extraction_text` list to verify whether each entry matches 
        one of the `valid_responses`. If an entry is invalid, it prints the index, the invalid entry, and the 
        corresponding pairwise comparison response for debugging purposes.

        Parameters:
            extraction_text (list): A list of extracted texts to validate.
            pairwise_comparison_responses (list): A list of pairwise comparison responses corresponding to 
                                                  the `extraction_text` entries, used for debugging.
            valid_responses (list): A list of valid responses that `extraction_text` entries are compared against.
        """
        for i in range(len(extraction_text)):
            if extraction_text[i] not in valid_responses:
                print(i)
                print(extraction_text[i])
                print(pairwise_comparison_responses[i])

    def create_win_df(self, pairwise_comparison_responses, extraction_results_final):
        """
        Creates a DataFrame summarizing pairwise text comparisons, concept-specific breakdowns, and outcomes.

        This method constructs a DataFrame that includes:
        - The IDs of the text pairs being compared.
        - The original texts (text1 and text2).
        - The concept-specific breakdowns for each text.
        - Binary outcome columns (`win1` and `win2`) indicating the results of the pairwise comparison.
        - A column for the pairwise comparison text.

        The outcomes are determined by the values in `extraction_results_final`:
        - `"Description 1"` assigns 1.0 to `win1` and 0.0 to `win2`.
        - `"Description 2"` assigns 0.0 to `win1` and 1.0 to `win2`.
        - `"Tie"` assigns 0.5 to both `win1` and `win2`.

        Parameters:
            pairwise_comparison_responses (list): A list of pairwise comparison texts corresponding to the comparisons.
            extraction_results_final (list): A list of comparison outcomes (e.g., "Description 1", "Description 2", or "Tie").

        Returns:
            df_final: A pd.DataFrame with the following columns:
                - `id`: IDs of the text pairs.
                - `win1`: Binary outcome for text1 (1.0, 0.0, or 0.5).
                - `win2`: Binary outcome for text2 (1.0, 0.0, or 0.5).
                - `text1`: The first text in the comparison.
                - `text2`: The second text in the comparison.
                - `csb1`: Concept-specific breakdown for the first text.
                - `csb2`: Concept-specific breakdown for the second text.
                - `comparison_text`: The pairwise comparison text.
        """
        id = []
        text1 = []
        text2 = []
        csb1 = []
        csb2 = []
        win1 = []
        win2 = []

        for i in range(len(self.ids)):
            id.append(self.ids[i])
            text1.append(self.id_text1_dict[self.ids[i]])
            text2.append(self.id_text2_dict[self.ids[i]])
            csb1.append(self.id_breakdown1_dict[self.ids[i]])
            csb2.append(self.id_breakdown2_dict[self.ids[i]])
            if extraction_results_final[i]=='Description 1':
                win1.append(1.0)
                win2.append(0.0)
            elif extraction_results_final[i]=='Description 2':
                win1.append(0.0)
                win2.append(1.0)
            elif extraction_results_final[i]=='Tie':
                win1.append(0.5)
                win2.append(0.5)

        df_final = pd.DataFrame({'id': id,
                                 'win1': win1,
                                 'win2': win2,
                                 'text1': text1,
                                 'text2': text2,
                                 'csb1': csb1,
                                 'csb2': csb2,
                                 'comparison_text': pairwise_comparison_responses})
        return df_final