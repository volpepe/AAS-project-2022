from collections import deque
import math
import numpy as np
from PIL import Image

from variables import PREPROCESSING_SIZE, PREPROCESSING_SIZE_RND, STACK_FRAMES

class State():
    def __init__(self, state_representation: np.ndarray) -> None:
        self.repr = state_representation

class StateManager():
    '''
    A utility object that pre-processes the frames and stacks the last 4 
    creating the new effective state for the agent.
    '''
    def __init__(self) -> None:
        # Create a screen buffer that holds the previous 4 frames. Every time we insert
        # a new frame, the old one gets discarded, creating a perfect tool for stacking
        # the last 4 frames
        self.image_size = PREPROCESSING_SIZE
        self.screen_buffer = deque([np.zeros(self.image_size) for _ in range(STACK_FRAMES)], maxlen=STACK_FRAMES)

    def preprocess_image(self, img) -> np.ndarray:
        '''
        Resizes the input image into the desired resolution and converts it in
        black and white
        '''
        return np.array(Image.fromarray(img).resize(self.image_size).convert('L'))

    def get_current_state(self, new_image) -> State:
        '''
        Method that gets as input the new image (received from the screen buffer
        of the game), preprocesses it and stacks it with the other 4 frames.
        It returns a State that is what the agent will use for the decision.
        '''
        # Preprocessing operation
        img = self.preprocess_image(new_image.transpose(1,2,0))     # 3x480x640 --> 480x640x3
        # Add the image to the buffer
        self.screen_buffer.append(img)
        # Stacks the images in the buffer and returns a single numpy array (stacking is done on last dimension)
        return State(np.stack(self.screen_buffer, axis=-1))

    def get_virtual_state(self, image) -> State:
        '''
        Like `get_current_state`, but does not append the image to the buffer
        and computes the frame stack without updating it. Useful for eg. getting
        state St+1.
        '''
        img = self.preprocess_image(image.transpose(1,2,0))
        return State(np.stack(list(self.screen_buffer[1:]) + [img], axis=1))


# Same class but specialized for RND
class RNDScreenPreprocessor():
    def __init__(self, min_clip:float=5.0, max_clip:float=5.0) -> None:
        # Create a screen buffer that holds the previous 4 frames. Every time we insert
        # a new frame, the old one gets discarded, creating a perfect tool for stacking
        # the last 4 frames
        self.preprocessed_screen_buffer = deque([np.zeros(PREPROCESSING_SIZE_RND) for _ in range(STACK_FRAMES)], maxlen=STACK_FRAMES)
        self.min_clip = min_clip
        self.max_clip = max_clip
        # We also keep a buffer of the last M frames without changes
        self.screen_buffer = deque([np.zeros((480,640)) for _ in range(M)], maxlen=M)

    def clear_preprocessed_screen_buffer(self):
        self.preprocessed_screen_buffer.clear()

    def append_new_screen(self, screen:np.ndarray):
        # For std and mean of last frame just append screen as is
        self.screen_buffer.append(screen)
        # For the policy net, we need to also preprocess the image
        img = self.preprocess_for_policy_net(screen)
        self.preprocessed_screen_buffer.append(img)  # Add pre-processed image to the buffer

    def get_state_for_policy_net(self, screen:np.ndarray):
        # Return stacked buffer (last 4 images)
        return np.stack(self.screen_buffer, axis=-1) 

    def get_virtual_state_for_policy_net(self, screen:np.ndarray):
        return np.stack(list(self.screen_buffer[1:]) + [self.preprocess_for_policy_net(screen)], axis=-1)

    def get_state_for_rnd(self, screen:np.ndarray):
        return self.preprocess_for_rnd(screen)      # Simply pre-process the screen

    def preprocess_for_policy_net(self, screen:np.ndarray):
        img = screen.transpose(1,2,0)                # 3x480x640 --> 480x640x3
        img = Image.fromarray(img)
        img = img.resize(PREPROCESSING_SIZE_RND)     # Resize
        img = img.convert('L')                       # To grayscale
        img = np.array(img) / 255                    # Normalize
        return img

    def preprocess_for_rnd(self, screen:np.ndarray):
        # No stacking, just standardization + clipping. We use the buffer to compute the running mean and std.
        last_frames = np.stack(self.screen_buffer, axis=-1)
        std_img = (screen - last_frames.mean()) / last_frames.std()
        return np.clip(std_img, self.min_clip, self.max_clip)