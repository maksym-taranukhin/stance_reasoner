import argparse
from datasets import load_from_disk
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from collections import Counter


from src.prompts import cot_prompt_template


llm_params = {
    "temperature": 0.7,
    "max_tokens": 150,
    # "callbacks": [RichStdOutCallbackHandler()],   # Uncomment this line to see the model's input and output
}
# llm = OpenAI(model="gpt-3.5-turbo-instruct", **llm_params)

llm = HuggingFacePipeline.from_model_id(
    model_id="llama-60B",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)


def main(args):
    ds = load_from_disk("data/preprocessed/" + args.dataset_name)["test"]

    prompt = ChatPromptTemplate.from_template(cot_prompt_template)
    cot_chain = (
        prompt
        | llm.bind(stop="\n\n")
        | StrOutputParser()
        | RunnableLambda(
            lambda res: dict(zip(["reasoning", "stance"], res.split("stance: ")))
        )
    )

    def self_consistency(x, n=3):
        completions = [
            cot_chain.invoke(dict(x)) for _ in range(n)
        ]

        reasonings = {f"reasoning_{i}": completions[i]["reasoning"].strip() for i in range(n)}
        stances = {f"stance_{i}": completions[i]["stance"].strip() for i in range(n)}
        pred = Counter(stances.values()).most_common(1)[0][0]
        confindence = Counter(stances.values()).most_common(1)[0][1] / n

        return {
            **x,
            **reasonings,
            **stances,
            "pred": pred,
            "confidence": confindence
        }

    ds = ds.map(lambda x: self_consistency(x))
    ds.save_to_disk("results" + args.dataset_name + "_cot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stance detection on tweets")
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the dataset", default="semeval2016"
    )

    args = parser.parse_args()
    main(args)
