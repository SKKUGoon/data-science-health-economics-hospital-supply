from typing import Optional

KCD_CODE_RANGE = {
    "KCD01": ('A00', 'B99'),
    "KCD02": ('C00', 'D48'),
    "KCD03": ('D50', 'D89'),
    "KCD04": ('E00', 'E90'),
    "KCD05": ('F00', 'F99'),
    "KCD06": ('G00', 'G99'),
    "KCD07": ('H00', 'H59'),
    "KCD08": ('H60', 'H95'),
    "KCD09": ('I00', 'I99'),
    "KCD10": ('J00', 'J99'),
    "KCD11": ('K00', 'K93'),
    "KCD12": ('L00', 'L99'),
    "KCD13": ('M00', 'M99'),
    "KCD14": ('N00', 'N99'),
    "KCD15": ('O00', 'O99'),
    "KCD16": ('P00', 'P96'),
    "KCD17": ('Q00', 'Q99'),
    "KCD18": ('R00', 'R99'),
    "KCD19": ('S00', 'T98'),
    "KCD20": ('U00', 'U99'),
    "KCD21": ('V01', 'Y98'),
    "KCD22": ('Z00', 'Z99'),
}

def _parse_kcd_code(s: str) -> tuple[str, int]:
    """
    Split the code into its alpha prefix and numeric part.
    Only the first three characters are used.
    """
    s = s.replace(" ", "").upper()[:3]
    alpha = ''
    num = ''
    for ch in s:
        if ch.isalpha():
            alpha += ch
        elif ch.isdigit():
            num += ch
    return alpha, int(num) if num else 0

def kcd_code_range_key(code: str) -> Optional[str]:
    """
    For a given code string, return the key of the KCD_CODE_RANGE dictionary whose range the code falls into.
    Only the first 3 characters of the code are considered.

    Args:
        code (str): KCD code string, e.g. 'A05', 'D609', etc.
    
    Returns:
        Optional[str]: The KCD range key the code falls into, or None if not found.
    """
    code = code.replace(" ", "").upper()
    code_fragment = code[:3]
    code_alpha, code_num = _parse_kcd_code(code_fragment)

    for range_key, (start, end) in KCD_CODE_RANGE.items():
        start_alpha, start_num = _parse_kcd_code(start)
        end_alpha, end_num = _parse_kcd_code(end)

        # Alphabetical comparison (inclusive)
        if start_alpha <= code_alpha <= end_alpha:
            # If alpha matches, check numeric range
            if code_alpha == start_alpha and code_alpha == end_alpha:
                if start_num <= code_num <= end_num:
                    return range_key
            elif code_alpha == start_alpha:
                if code_num >= start_num:
                    return range_key
            elif code_alpha == end_alpha:
                if code_num <= end_num:
                    return range_key
            else:
                # code_alpha is strictly between start_alpha and end_alpha
                return range_key

    return None
