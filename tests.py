from botsort.botsort import BotSort
from botsort.botsort.config import BotSortConfig, SparseOptFlowConfig
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
from typing import Optional, Self, overload

class Boxes:
    """
    Stored Data for bounding boxes per detector's forward call.\n
    data: float torch.Tensor or np.ndarray. Shape (n, 6|7).
          \t\t[n, :4] contains bounding box coordinates xyxy format.\n
          \t\t[n, 5] contains each class id.\n
          \t\t[n, 6] contains each confidence score.\n
          \t\t[n, 7] (if available) contains each track id.\n
    data is generally accessed by thier associated property.\n
    original_height, original width: int, int.\n
        \t\tThe original height width of the image. 
        Used to normalize bounding box coordinates.
    
    """
    def __init__(
        self,
        data: torch.Tensor | np.ndarray,
        original_height: int,
        original_width: int,
    ):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        else:
            data = np.asarray(data)
        
        assert data.ndim == 2 and data.shape[-1] in {6,7}

        data[:, :4:2] = data[:, :4:2].clip(min=0, max=original_width)
        data[:, 1:4:2] = data[:, 1:4:2].clip(min=0, max=original_height)
        data[:, :2] = np.floor(data[:, :2])
        data[:, 2:4] = np.ceil(data[:, 2:4])

        self.data = data.astype(np.float32)
        self.has_track = (data.shape[-1] == 7 and self.track_id != -1)
        self.original_height = original_height
        self.original_width = original_width
        self.orig_shape = tuple([original_height, original_width])
        
    @classmethod
    def from_mask(
        cls,
        masks: np.ndarray | torch.Tensor,
        scores: np.ndarray | torch.Tensor,
        classes: np.ndarray | torch.Tensor,
        tracks: Optional[np.ndarray | torch.Tensor] = None,
    ):
        scores = np.asarray(scores)
        classes = np.asarray(classes)
        
        masks = torch.as_tensor(masks)
        boxes = masks_to_boxes(masks)
        
        boxes = np.asarray(boxes)
        if classes.ndim == 1:
            classes = classes[:, None]
        if scores.ndim == 1:
            scores = scores[:, None]
        
        data = np.concatenate([boxes, classes, scores], axis=-1)# (n, 6)
        if tracks is not None:
            tracks = np.asarray(tracks)
            if tracks.ndim == 1:
                tracks = tracks[:, None]
            data = np.concatenate([data, tracks], axis=-1)#(n,7)

        return cls(data.astype(np.float32), masks.shape[-2], masks.shape[-1])
        
        
    def __len__(self):
        return self.data.shape[0]
    
    def __repr__(self):
        str_repr = ""

        for box in self:
            str_repr += "Bounding Box: " + " ".join(str(x) for x in box.xyxy.flatten())
            str_repr += "\n\t"
            str_repr += "Class Id: " + " ".join(str(x) for x in box.cls.flatten())
            str_repr += "\n\t"
            str_repr += "Score: " + " ".join(str(x) for x in box.conf.flatten())
            
            if box.has_track:
                str_repr += "\n\t"
                str_repr += "Track ID: " + " ".join(str(x) for x in box.track_id.flatten())
            
            str_repr += "\n"

        return str_repr.strip()
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if data.ndim == 1:
            data = data[None, :]
        return self.__class__(data, self.original_height, self.original_width)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def xyxy(self):
        return self.data[:, :4]
    
    @property
    def xywh(self):
        "Bounding box defined by top left xy, and width, height"
        
        _xywh = np.empty_like(self.xyxy)
        _xywh[:, 0] = self.data[:, 0]
        _xywh[:, 1] = self.data[:, 1]
        _xywh[:, 2] = self.data[:, 2] - self.data[:, 0]
        _xywh[:, 3] = self.data[:, 3] - self.data[:, 1]

        return _xywh

    @property
    def cxywh(self):
        "Bounding box defined by center xy, and width, height"

        xywh = self.xywh
        xywh[:, 0] = self.data[:, 0] + (xywh[:, 2]/2)
        xywh[:, 1] = self.data[:, 1] + (xywh[:, 3]/2)

        return xywh
    
    @property
    def xyxyn(self):
        data = self.data.copy()
        data[:, [2,0]] /= self.orig_shape[1]
        data[:, [3,1]] /= self.orig_shape[0]

        return data[:, :4]
    
    @property
    def cxywhn(self):

        data = self.cxywh
        data[:, [2,0]] /= self.orig_shape[1]
        data[:, [3,1]] /= self.orig_shape[0]

        return data[:, :4]
    
    @property
    def cls(self):
        return self.data[:, 4]
    
    @property
    def conf(self):
        return self.data[:, 5]
    
    @property
    def track_id(self):
        return self.data[:, 6]
    
    @overload
    def update(self, new_boxes: Self) -> None:
        """
        Update the base data for a specific set of bounding boxes
        
        :param new_boxes: The other instance of Boxes. 
            Useful for pre verification before updating
        :type new_boxes: Self
        """
        ...
    
    @overload
    def update(self, new_boxes: torch.Tensor | np.ndarray) -> None:
        """
        
        Update the base data for a specific set of bounding boxes

        :param new_boxes: A Torch tensor or numpy array.
            It should have the same shape as the other boxes with the same xyxy data

        """
        ...
    
    @overload
    def update(
        self,
        new_boxes: torch.Tensor | np.ndarray,
        cls_id: list[int | float],
        score: list[int | float],
        track_id: list[int | float] | None = None,
    ) -> None:
        """
        Docstring for update
        
        :param new_boxes: A list of bounding boxes in xyxy format
        :type new_boxes: torch.Tensor | np.ndarray
        :param cls_id: A list of class ids equal length to the number of boxes
        :type cls_id: list[int | float]
        :param score: A list of confidence scores equal length to the number of boxes
        :type score: list[int | float]
        :param track_id: A list of ids for tracking equal length to the number of boxes.
            Where no track was found, -1 should be used.
        :type track_id: list[int | float] | None
        """
        ...
    
    def update(
        self,
        new_boxes: Self | torch.Tensor | np.ndarray,
        cls_id: list[int | float] | None = None,
        score: list[int | float] | None = None,
        track_id: list[int | float] | None = None,
    ) -> None:

        bounding_boxes = new_boxes
        if isinstance(bounding_boxes, self.__class__):
            bounding_boxes = bounding_boxes.data
        elif isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = bounding_boxes.numpy()
        elif not isinstance(bounding_boxes, np.ndarray):
            raise TypeError("new_boxes has incorrect type")
        
        if bounding_boxes.ndim == 1:
            bounding_boxes = bounding_boxes[None, :]
        
        if bounding_boxes.shape[-1] == 4:
            assert cls_id is not None, f"Class id should not be None"
            assert score is not None, f"Confidence Score should not be None"
            additions = [cls_id, score] + ([track_id] if track_id is not None else [])
            bounding_boxes = np.concat([bounding_boxes, np.asarray(additions)], axis=-1)
        
        assert bounding_boxes.shape[-1] < 8, f"Incorrect data shape {bounding_boxes.shape}"
        new_boxes = self.__class__(bounding_boxes, self.original_height, self.original_width)
        
        # check if values were in the same format during update.
        # Mainly for numpy and torch.
        if new_boxes.data.shape[0] == 0 and track_id is not None:
            raise RuntimeError("No boxes were given when tracks were ready to update")
        elif (
            not np.allclose(self.xyxy, new_boxes.xyxy, rtol=1, atol=4)
            and len(bounding_boxes) != len(self)
        ):
            raise RuntimeError("Bounding box mismatch. " + \
            f"Expected {self.xyxy} found {new_boxes.xyxy}")

        self.data = new_boxes.data


class Tracker(BotSort):

    def update(
        self,
        img: np.ndarray,
        detections: Boxes
    ) -> Boxes:
        """
        Match Boxes from any detection or segementation model to thier 
        tracks. If no tracks, an empty Boxes instances is returned
        
        :param img: Original image used for the example, in CHW format. 
            Only the Height and Width are currently used.
        :type img: np.ndarray
        :param detections: A combined list of Boxes from a single example
        :type detections: Boxes
        :return: Boxes with an additional track id, or an empty instance.
        :rtype: Boxes
        """

        if self.tracker is None:
            raise RuntimeError(f"must init tracker before using it")
        
        tracks = self.track(
            detections.xywh,
            detections.conf,
            detections.cls,
            img
        )
        data = np.empty(shape=(0, 6), dtype=np.float32)
        
        if len(tracks) != 0:
            # drop index (last value) since we assume a batch size of 1
            data = tracks
            data[:, 2] = data[:, 0] + data[:, 2]
            data[:, 3] = data[:, 1] + data[:, 3]
            #data = np.concatenate([detections.data, tracks[None, :]], dtype=np.float32, axis=1)
        
        boxes = Boxes(
            data=data,
            original_height=img.shape[-1],
            original_width=img.shape[-2]
        )

        return boxes
    
    def __call__(
        self,
        img: np.ndarray,
        detections: Boxes
    ) -> Boxes:
        
        return self.update(img, detections)

FPS = 30
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

frame = np.zeros((3,640,640))
# State 1: Never Tracklet
tracker.init_tracker(FPS)

box_tlwh = [100, 100, 150, 150]  # x, y, x, y
scores = [0.2]                  # Below _track_low_thresh (0.3)
class_ids = [1]
data = np.asarray([box_tlwh + class_ids + scores])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)

tracker.update(frame, boxes)
# State 2: Low-Confidence Matched
## In 2nd Association

tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Create a track first
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.8]               # High confidence
class_ids_f1 = [1]

data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("First Frame")
print(tracker.update(frame, boxes))

# Frame 2: Low confidence detection nearby
box_tlwh_f2 = [105, 105, 155, 155] 
scores_f2 = [0.4]
class_ids_f2 = [1]

data = np.asarray([box_tlwh_f2 + class_ids_f2 + scores_f2])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("Second Frame")
print(tracker.update(frame, boxes))

## Unmatched
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Create a track
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.8]
class_ids_f1 = [1]


data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("First Frame")
print(tracker.update(frame, boxes))


# Frame 2: Low confidence detection far away
box_tlwh_f2 = [400, 300, 450, 350]  # Far from existing track
scores_f2 = [0.45]                 # Low confidence
class_ids_f2 = [1]

data = np.asarray([box_tlwh_f2 + class_ids_f2 + scores_f2])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("Second Frame")
print(tracker.update(frame, boxes))
# State 3: High-Conf
## Matched
### Tracked Track
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Create a track
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.85]
class_ids_f1 = [1]

data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("First Frame")
print(tracker.update(frame, boxes))

# Frame 2: High confidence detection nearby
box_tlwh_f2 = [102, 102, 152, 152]  # Slight movement
scores_f2 = [0.8]                  # High confidence
class_ids_f2 = [1]

data = np.asarray([box_tlwh_f2 + class_ids_f2 + scores_f2])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("Second Frame")
print(tracker.update(frame, boxes))

### Lost Track
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Create a track
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.85]
class_ids_f1 = [1]

data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("First Frame")
print(tracker.update(frame, boxes))

# Frame 2: No detection (track becomes lost)
box_tlwh_f2 = []
scores_f2 = []
class_ids_f2 = []

data = np.asarray([box_tlwh_f2 + class_ids_f2 + scores_f2])

boxes = Boxes(
    data=data.reshape((0,6)),
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("Second Frame")
print(tracker.update(frame, boxes))

# Frame 3: High-conf detection reappears
box_tlwh_f3 = [105, 105, 155, 155]
scores_f3 = [0.82]
class_ids_f3 = [1]

data = np.asarray([box_tlwh_f3 + class_ids_f3 + scores_f3])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("Third Frame")
print(tracker.update(frame, boxes))

## Unmatched
### Unconfirmed Track
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Create unconfirmed track
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.71]
class_ids_f1 = [1]

data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("First Frame")
print(tracker.update(frame, boxes))

# Frame 2: Another high-conf detection nearby, but existing track unconfirmed
box_tlwh_f2 = np.asarray([
    [300, 300, 350, 350],   # New detection
    [103, 103, 153, 153]   # Near the unconfirmed track
])
scores_f2 = np.asarray([
    [0.65],
    [0.75]
])
class_ids_f2 = np.asarray([
    [2],
    [1]
])

data = np.concat([box_tlwh_f2, class_ids_f2, scores_f2], axis=-1)

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)

print("Second Frame")
print(tracker.update(frame, boxes))
### Umatched, score over new track thresh
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Empty or track elsewhere
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.8]
class_ids_f1 = [1]

data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("\n\nFirst Frame")
print(tracker.update(frame, boxes))

# Frame 2: New high-conf detection far away
box_tlwh_f2 = np.asarray([
    [100, 100, 150, 150],   # Original track position
    [400, 300, 460, 360]   # NEW detection far away
])
scores_f2 = np.asarray([
    [0.79],
    [0.99]
])
class_ids_f2 = np.asarray([
    [1],
    [2]
])
data = np.concat([box_tlwh_f2, class_ids_f2, scores_f2], axis=-1)

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)
print("Boxes")
print(boxes)
print("\nSecond Frame")
print(tracker.update(frame, boxes))

### Orphaned
tracker = Tracker(
    BotSortConfig(
        track_low_thresh=0.3,
        match_thresh=0.8,
        proximity_thresh=0.8,
        #frame_rate=30,
        #track_buffer=30,
    ),
    SparseOptFlowConfig(),
    enable_gmc=False
)

tracker.init_tracker(FPS)

# Frame 1: Create existing track
box_tlwh_f1 = [100, 100, 150, 150]
scores_f1 = [0.85]
class_ids_f1 = [1]

data = np.asarray([box_tlwh_f1 + class_ids_f1 + scores_f1])

boxes = Boxes(
    data=data,
    original_height=frame.shape[-2],
    original_width=frame.shape[-1]
)
print("First Frame")
print(tracker.update(frame, boxes))

# Frame 2: Detection that's high-conf but below new track threshold
box_tlwh_f2 = np.asarray([
    [100, 100, 150, 150],   # Matches existing
    [400, 300, 460, 360]   # NEW area, borderline score
])
scores_f2 = np.asarray([
    [0.82],
    [0.65]
])
class_ids_f2 = np.asarray([
    [1],
    [2]
])

data = np.concat([box_tlwh_f2, class_ids_f2, scores_f2], axis=-1)

boxes = Boxes(
    data=data,
    original_height=frame.shape[-1],
    original_width=frame.shape[-2]
)

print("Second Frame")
print(tracker.update(frame, boxes))
