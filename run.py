import onnxruntime_genai as og
import argparse
import time

def main(args):
    print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'cpu-int4-rtn-block-32-acc-level-4')
    print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    print("Tokenizer created")
    print()
    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    
    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length
    if 'max_length' not in search_options:
        search_options['max_length'] = 2048*8

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    # Keep asking for input prompts in a loop
    while True:
        content = ""
        while not content:
            try:
                with open(input("Entrez le chemin du fichier: "), 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                print("Le fichier n'a pas été trouvé.")
            except Exception as e:
                print(f"Une erreur est survenue : {e}")
        
        #text = "Voici un document au format HTML : \n"
        #text += content
        
        text += input("Requête: ")

        started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
        print("Generator created")

        print("Running generation loop ...")

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', default=False, help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
    args = parser.parse_args()
    main(args)
