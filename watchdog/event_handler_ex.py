from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import time
import os
from datetime import datetime
from watchdog_fileobserver_ex import main


def create_directory(file_path=None):
    # Get the current date in the format of 'year-month-day'
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Create a folder with the current date
    folder_path = f'{file_path}/{current_date}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path
    else:
        return folder_path


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        dir_path = event.src_path.split('/input_files')
        processed_files = f'{dir_path[0]}/processed_files'

        child_processed_dir = create_directory(file_path=processed_files)

        if event:
            print("file created:{}".format(event.src_path))
            # call function here
            main(file_name=event.src_path)

            file_name = event.src_path.split('/')[-1]
            destination_path = f'{child_processed_dir}/{file_name}'

            shutil.move(event.src_path, destination_path)
            print("file moved:{} to {}".format(event.src_path, destination_path))


if __name__ == "__main__":
    observer = Observer()
    event_handler = MyHandler()
    observer.schedule(event_handler, path='./input_files', recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(300)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()