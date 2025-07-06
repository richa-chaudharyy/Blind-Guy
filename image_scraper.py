from icrawler.builtin import BingImageCrawler
import os

def download_reference_images(object_name, num_images=5):
    save_dir = f"reference_images/{object_name}"
    os.makedirs(save_dir, exist_ok=True)
    crawler = BingImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=object_name, max_num=num_images)

if __name__ == "__main__":
    object_name = "cane stick"
    download_reference_images(object_name)
