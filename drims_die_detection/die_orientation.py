"""
die_orientation.py
===================
Encodes 3D d6 die orientation and maps all 6 face pip values based on top face,
lateral face detections, and standard right-handed die geometry (opposite sum = 7).
"""

from __future__ import annotations

# Standard Right-Handed D6 Die Lookup:
# Maps (top_pips, front_x_pips) -> right_y_pips
# Opposite faces sum to 7. Right-handed chirality rule: 1-2-3 meet CCW around corner.
_RIGHT_HAND_D6_RIGHT_FACE: dict[tuple[int, int], int] = {
    # Top = 1 (Bottom = 6)
    (1, 2): 3, (1, 3): 5, (1, 5): 4, (1, 4): 2,
    # Top = 6 (Bottom = 1)
    (6, 2): 4, (6, 4): 5, (6, 5): 3, (6, 3): 2,
    # Top = 2 (Bottom = 5)
    (2, 1): 4, (2, 4): 6, (2, 6): 3, (2, 3): 1,
    # Top = 5 (Bottom = 2)
    (5, 1): 3, (5, 3): 6, (5, 6): 4, (5, 4): 1,
    # Top = 3 (Bottom = 4)
    (3, 1): 2, (3, 2): 6, (3, 6): 5, (3, 5): 1,
    # Top = 4 (Bottom = 3)
    (4, 1): 5, (4, 5): 6, (4, 6): 2, (4, 2): 1,
}


def resolve_die_orientation(
    top_pips: int,
    x_pos_pips: int | None = None,
    y_pos_pips: int | None = None,
) -> dict:
    """Resolve the 6-face pip mapping of a standard right-handed d6 die.

    Parameters
    ----------
    top_pips : int (1..6)
        Pip count on the top face (+Z direction).
    x_pos_pips : int | None (1..6)
        Pip count on the primary lateral face (+X direction).
    y_pos_pips : int | None (1..6)
        Pip count on the secondary lateral face (+Y direction).

    Returns
    -------
    dict with keys:
      is_fully_determined : bool
      top_pips            : int (+Z)
      bottom_pips         : int (-Z = 7 - top_pips)
      x_pos_pips          : int | None (+X)
      x_neg_pips          : int | None (-X = 7 - x_pos_pips)
      y_pos_pips          : int | None (+Y)
      y_neg_pips          : int | None (-Y = 7 - y_pos_pips)
      face_map            : dict mapping "+Z", "-Z", "+X", "-X", "+Y", "-Y" to pip counts
    """
    bot_pips = 7 - top_pips if 1 <= top_pips <= 6 else None

    if x_pos_pips is not None and 1 <= x_pos_pips <= 6:
        x_neg_pips = 7 - x_pos_pips
        if y_pos_pips is None and top_pips is not None:
            y_pos_pips = _RIGHT_HAND_D6_RIGHT_FACE.get((top_pips, x_pos_pips))

        if y_pos_pips is not None:
            y_neg_pips = 7 - y_pos_pips
            fully_determined = True
        else:
            y_neg_pips = None
            fully_determined = False
    else:
        x_neg_pips = None
        y_neg_pips = None
        fully_determined = False

    face_map = {
        "+Z": top_pips,
        "-Z": bot_pips,
        "+X": x_pos_pips,
        "-X": x_neg_pips,
        "+Y": y_pos_pips,
        "-Y": y_neg_pips,
    }

    return {
        "is_fully_determined": fully_determined,
        "top_pips": top_pips,
        "bottom_pips": bot_pips,
        "x_pos_pips": x_pos_pips,
        "x_neg_pips": x_neg_pips,
        "y_pos_pips": y_pos_pips,
        "y_neg_pips": y_neg_pips,
        "face_map": face_map,
    }
