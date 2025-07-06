import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_object_name(input_text):
    """
    Extract object keyword from text using NLP.
    If NLP fails, fallback to full input text.
    """
    doc = nlp(input_text)

    # Extract nouns (objects usually are nouns)
    objects = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

    if objects:
        object_found = objects[0]  # Take first noun for now
        print(f"Extracted Object (NLP): {object_found}")
        return object_found
    else:
        # Fallback: use entire sentence
        object_fallback = input_text.strip().lower().replace(".", "").replace(",", "")
        print(f"No noun found. Fallback Object: {object_fallback}")
        return object_fallback

# Sample test
if __name__ == "__main__":
    sample_text = "headphone"
    result = extract_object_name(sample_text)
    print("Object:", result)
