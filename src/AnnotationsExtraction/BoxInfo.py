class BoxInfo:
    """
    Represents parsed annotation data for a single bounding box.

    Parameters
    ----------
    line : str
        A whitespace-separated string containing:
        [playerId, xMin, yMin, xMax, yMax, frameId, lost, grouping, generated, category]

    Attributes
    ----------
    player_id : str
        Identifier for the player.
    category : str
        Category or label of the object.
    x_min, y_min, x_max, y_max : int
        Bounding box coordinates.
    frame_id : str
        Frame identifier.
    lost, grouping, generated : int
        Status flags for the bounding box.
    """

    def __init__(self, line: str):
        parts = line.strip().split()
        if len(parts) < 10:
            raise ValueError(f"Invalid annotation format: expected 10 parts, got {len(parts)}")

        (
            self.player_id,
            x_min, y_min, x_max, y_max,
            self.frame_id,
            lost, grouping, generated,
            self.category
        ) = parts

        # Convert numeric values safely
        try:
            self.x_min = int(x_min)
            self.y_min = int(y_min)
            self.x_max = int(x_max)
            self.y_max = int(y_max)
            self.lost = int(lost)
            self.grouping = int(grouping)
            self.generated = int(generated)
        except ValueError as e:
            raise ValueError(f"Invalid numeric field in line: {e}")

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Returns the bounding box as a tuple (x_min, y_min, x_max, y_max)."""
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def is_valid(self) -> bool:
        """Checks if the bounding box has valid coordinates."""
        return self.x_min < self.x_max and self.y_min < self.y_max

    def __repr__(self) -> str:
        return (
            f"BoxInfo(player_id={self.player_id}, frame_id={self.frame_id}, "
            f"category={self.category}, box={self.box}, lost={self.lost})"
        )
