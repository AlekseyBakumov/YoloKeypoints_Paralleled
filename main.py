import argparse
import cv2
import time
from threading import Thread, Lock
from queue import Queue
from ultralytics import YOLO
import numpy as np

class VideoProcessor:
    def __init__(self, video_path, output_path, model_path='yolov8s-pose.pt', num_threads=4, frame_size=(640, 480)):
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        self.num_threads = num_threads
        self.frame_size = frame_size
        self.frame_queue = Queue(maxsize=100)
        self.processed_frames = {}
        self.lock = Lock()
        self.running = True
        self.frame_count = -1
        self.current_frame = 0
        self.fps = 30
        self.start_time = 0
        self.writer_thread = None
        self.reader_thread = None
        self.workers = []

    def load_model(self):
        """Load YOLOv8 pose estimation model"""
        model = None
        try:
            model = YOLO(self.model_path)
            print(f"Loaded YOLOv8 model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        return model

    def video_reader(self):
        """Read frames from video file and put them in the queue"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file {self.video_path}")

            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing video: {self.video_path}")
            print(f"Total frames: {self.frame_count}, FPS: {self.fps}")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame if needed
                if (frame.shape[1], frame.shape[0]) != self.frame_size:
                    frame = cv2.resize(frame, self.frame_size)

                # Put frame in queue with its index
                self.frame_queue.put((self.current_frame, frame))
                self.current_frame += 1

            cap.release()
            print("Finished reading video frames")
            
            # Signal workers to finish
            for _ in range(self.num_threads):
                self.frame_queue.put((None, None))

        except Exception as e:
            print(f"Error in video reader: {e}")
            self.running = False

    def process_frame(self, thread_id):
        """Worker thread function to process frames"""
        print(f"Thread {thread_id} started")

        model = self.load_model()

        while self.running:
            frame_idx, frame = self.frame_queue.get()
            
            # Check for termination signal
            if frame is None:
                self.frame_queue.task_done()
                break

            try:
                # Process frame with YOLOv8
                #print(f"Thread {thread_id} processing frame {frame_idx}")
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()

                # Store processed frame
                with self.lock:
                    self.processed_frames[frame_idx] = annotated_frame

            except Exception as e:
                print(f"Thread {thread_id} error processing frame {frame_idx}: {e}")
                with self.lock:
                    self.processed_frames[frame_idx] = frame.copy()  # Store original if error

            self.frame_queue.task_done()

        print(f"Thread {thread_id} finished")

    def video_writer(self):
        """Write processed frames to output video"""
        try:
            # Ensure the output path has .mp4 extension
            if not self.output_path.lower().endswith('.mp4'):
                self.output_path += '.mp4'

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            if not out.isOpened():
                raise IOError(f"Could not open video writer for {self.output_path}")

            written_frames = 0
            last_print = 0

            while (written_frames < self.frame_count and self.running) or (self.current_frame == -1):
                if written_frames in self.processed_frames:
                    with self.lock:
                        frame = self.processed_frames.pop(written_frames)
                    out.write(frame)
                    written_frames += 1
                    
                    # Print progress every 5 seconds
                    if time.time() - last_print > 5:
                        progress = (written_frames / self.frame_count) * 100
                        print(f"Progress: {progress:.1f}% ({written_frames}/{self.frame_count} frames)")
                        last_print = time.time()
                else:
                    time.sleep(0.01)

            out.release()
            print(f"Finished writing output to {self.output_path}")

        except Exception as e:
            print(f"Error in video writer: {e}")
            self.running = False
        finally:
            if 'out' in locals() and out.isOpened():
                out.release()

    def run(self):
        """Main processing function"""
        self.start_time = time.time()
        
        try:
            # Load model
            #self.load_model()

            self.reader_thread = Thread(target=self.video_reader)
            self.reader_thread.start()

            self.workers = []
            for i in range(self.num_threads):
                worker = Thread(target=self.process_frame, args=(i,))
                worker.start()
                self.workers.append(worker)

            self.writer_thread = Thread(target=self.video_writer)
            self.writer_thread.start()

            print("Stopping workers...")
            self.reader_thread.join()
            for worker in self.workers:
                worker.join()
            
            self.stop()

            print("Stopping writer...")
            self.writer_thread.join()

            processing_time = time.time() - self.start_time
            print(f"\nProcessing completed in {processing_time:.2f} seconds")
            print(f"Average FPS: {self.frame_count / processing_time:.2f}")

        except KeyboardInterrupt:
            print("\nReceived interrupt signal, shutting down...")
            self.running = False
            if self.writer_thread:
                self.writer_thread.join()
            raise

        except Exception as e:
            print(f"Error in main processing: {e}")
            self.running = False
            raise

    def stop(self):
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation on Video with Multi-threading")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_name", help="Name for output video file")
    parser.add_argument("--threads", type=int, default=4, 
                       help="Number of threads to use (default: 4)")
    parser.add_argument("--frame_size", type=str, default="640x480",
                       help="Frame size in format WxH (default: 640x480)")

    args = parser.parse_args()

    try:
        # Parse frame size
        width, height = map(int, args.frame_size.split('x'))
        
        # Create processor instance
        processor = VideoProcessor(
            video_path=args.video_path,
            output_path=args.output_name,
            num_threads=args.threads,
            frame_size=(width, height)
        )
        processor.run()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
