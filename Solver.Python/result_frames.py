from dataclasses import dataclass
from result_data import result_data

@dataclass
class result_frames:
    def __init__(self, u_frames, f, has_u_t= False, has_derivs= False, has_laplacian= False):
        self.u_frames = u_frames
        self.num_frames = len(u_frames)
        self.frame_contains_result_data = all(isinstance(elem, result_data) for elem in u_frames)
        self.f = f
        self.has_u_t = has_u_t
        self.has_derivs = has_derivs
        self.has_laplacian = has_laplacian

    def get_u_frames(self):
        if self.frame_contains_result_data:
            return [frame.u for frame in self.u_frames]
        else:
            return None
        
    def get_u_t_frames(self):
        if self.frame_contains_result_data:
            return [frame.u_t for frame in self.u_frames]
        else:
            return None

    def get_u_x_frames(self):
        if self.frame_contains_result_data:
            return [frame.u_x for frame in self.u_frames]
        else:
            return None

    def get_u_y_frames(self):
        if self.frame_contains_result_data:
            return [frame.u_y for frame in self.u_frames]
        else:
            return None

    def get_laplacians(self):
        if self.frame_contains_result_data:
            return [frame.laplacian for frame in self.u_frames]
        else:
            return None