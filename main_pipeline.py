from speech_google import speech_to_text_google as speech_to_text
from nlp_extractor import extract_object_name
from image_scraper import download_reference_images
from object_navigation import object_navigation_mode
from object_training import train_custom_model
from navigation_assistant import navigation_mode
from constants import YOLO_CLASSES
from speech_utils import speak

# Import or define the speak function

# Add YOLO_CLASSES here:
from constants import YOLO_CLASSES

def object_in_yolo(object_name):
    object_name = object_name.lower()
    return object_name in YOLO_CLASSES

def main():
    print("\n==== BLIND ASSISTIVE SYSTEM STARTED ====\n")
    
    speak("Please select mode. Say Indoor for object navigation or Outdoor for obstacle navigation.")

    mode_text = speech_to_text()

    if mode_text is None:
        speak("Sorry, could not recognize speech. Exiting.")
        return

    mode_text = mode_text.lower()

    if "indoor" in mode_text:
        speak("Indoor mode selected. Please say the object name you are looking for.")
        spoken_text = speech_to_text()
        if not spoken_text:
            speak("Speech recognition failed. Exiting.")
            return

        object_name = extract_object_name(spoken_text)
        if not object_name:
            speak("No object extracted. Exiting.")
            return

        speak(f"Searching for {object_name}")

        # Here's the new logic
        if object_in_yolo(object_name):
            speak(f"{object_name} found in YOLO default dataset.")
            object_navigation_mode(object_name, pretrained=True)
        else:
            speak(f"{object_name} not found in YOLO. Starting image scraping.")
            download_reference_images(object_name, num_images=5)

            # Call your training pipeline (auto-training)
            train_custom_model(object_name)

            # Then detect using custom model
            object_navigation_mode(object_name, pretrained=False)

    elif "outdoor" in mode_text:
        speak("Outdoor mode selected. Starting obstacle detection.")
        navigation_mode()

    else:
        speak("Mode not recognized. Exiting.")

if __name__ == "__main__":
    main()
