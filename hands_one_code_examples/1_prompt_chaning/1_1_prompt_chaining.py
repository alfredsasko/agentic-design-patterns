import json
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_chain(llm: ChatOpenAI):
	prompt_extract = ChatPromptTemplate.from_template(
		"Extract the technical specifications from the following text:\n\n{text_input}"
	)

	prompt_transform = ChatPromptTemplate.from_template(
		"Transform the following specifications into a JSON object with "
		"'cpu', 'memory', and 'storage' as keys:\n\n{specifications}. "
		"Return only valid JSON."
	)

	extraction_chain = prompt_extract | llm | StrOutputParser()

	full_chain = (
		{"specifications": extraction_chain}
		| prompt_transform
		| llm
		| StrOutputParser()
	)

	return full_chain


def validate_json_structure(result_text: str):
	try:
		parsed = json.loads(result_text)
	except json.JSONDecodeError as error:
		return False, None, f"Output is not valid JSON: {error}"

	expected_keys = {"cpu", "memory", "storage"}
	actual_keys = set(parsed.keys())
	if actual_keys != expected_keys:
		return (
			False,
			parsed,
			f"Unexpected keys. Expected {expected_keys}, got {actual_keys}",
		)

	return True, parsed, "JSON structure is valid."


def validate_expected_values(parsed: dict, expected_values: dict):
	for key, expected_value in expected_values.items():
		actual_value = str(parsed.get(key, "")).lower()

		expected_tokens = (
			expected_value if isinstance(expected_value, list) else [expected_value]
		)

		for token in expected_tokens:
			if token.lower() not in actual_value:
				return (
					False,
					f"Value mismatch for '{key}'. Expected to include '{token}', got '{parsed.get(key)}'",
				)

	return True, "Values look correct."


def main():
	load_dotenv()

	if not os.getenv("OPENAI_API_KEY"):
		raise EnvironmentError(
			"OPENAI_API_KEY is not set. Add it to your .env file before running this script."
		)

	llm = ChatOpenAI(temperature=0)
	full_chain = build_chain(llm)

	input_text = (
		"The new laptop model features a 3.5 GHz octa-core processor, "
		"16GB of RAM, and a 1TB NVMe SSD."
	)

	final_result = full_chain.invoke({"text_input": input_text})

	print("\n--- Final JSON Output ---")
	print(final_result)

	structure_ok, parsed, structure_message = validate_json_structure(final_result)
	print(f"\nStructure validation: {structure_message}")

	if structure_ok and parsed is not None:
		expected = {
			"cpu": ["3.5 ghz", "octa-core"],
			"memory": "16GB",
			"storage": "1TB NVMe SSD",
		}
		values_ok, values_message = validate_expected_values(parsed, expected)
		print(f"Value validation: {values_message}")
		if not values_ok:
			raise ValueError(values_message)
	else:
		raise ValueError(structure_message)


if __name__ == "__main__":
	main()
