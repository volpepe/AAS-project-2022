from collections import deque
import numpy as np
from PIL import Image

from variables import PREPROCESSING_SIZE, STACK_FRAMES

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
        self.screen_buffer = deque([np.zeros(self.image_size) for _ in range(STACK_FRAMES)])

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
        # Stacks the images in the buffer and returns a single numpy array
        return State(np.stack(self.screen_buffer))