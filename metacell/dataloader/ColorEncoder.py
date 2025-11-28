class CellTypeMarkerEncoder(object):
    def __init__(self, marker_df=None, default_color="#DCDCDC", combined_color="#dc143c"):
        self.marker_map = {}
        self.color_map = {}
        self.default_color = default_color
        self.combined_color = combined_color

        # Initialize marker map and color map using the provided DataFrame
        if marker_df is not None:
            for index, row in marker_df.iterrows():
                marker = row['marker_name']
                color_code = row['color_code']
                bit_value = 1 << len(self.marker_map)
                self.marker_map[marker] = bit_value
                self.color_map[bit_value] = color_code

    def encode(self, markers):
        if isinstance(markers, str):
            markers = [markers]
        encoding = 0
        for marker in markers:
            if marker in self.marker_map:
                encoding |= self.marker_map[marker]
            else:
                raise ValueError(f"Unknown marker: {marker}")
        return encoding

    def decode(self, encoding):
        # Calculate the max possible encoding value.
        max_encoding = sum(self.marker_map.values())

        # Check if the encoding is within the valid range
        if encoding > max_encoding:
            raise ValueError(f"Invalid encoding value: {encoding}. Maximum valid value is {max_encoding}.")

        decoded_markers = []
        for marker, bit in self.marker_map.items():
            if encoding & bit:
                decoded_markers.append(marker)
        return decoded_markers

    def add_marker(self, marker, color_code):
        if marker in self.marker_map:
            raise ValueError(f"Marker {marker} already exists.")
        bit_value = 1 << len(self.marker_map)
        self.marker_map[marker] = bit_value
        self.color_map[bit_value] = color_code

    def available_markers(self):
        return self.marker_map

    def encode_to_color(self, encoding):
        # If the encoding has a color assigned, return it
        if encoding in self.color_map:
            return self.color_map[encoding]
        elif encoding == 0:
            return self.default_color
        elif bin(encoding).count('1') > 1:
            return self.combined_color

    def encode_to_name(self, encoding):
        if encoding == 0:
            return "Unknown"
        for marker, bit_value in self.marker_map.items():
            if encoding == bit_value:
                return marker
        return "Mixed"