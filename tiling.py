import os
from pathlib import Path

from PIL import Image


class Tiler:

    def __init__(self, net_size: tuple, srcdir: str = '.', destdir: str = '.'):
        """
        """
        self.srcdir = Path(srcdir)
        self.destdir = Path(destdir)
        self.img = None
        self.net_width = net_size[0]
        self.net_height = net_size[1]
        print(f'net size: {self.net_width} x {self.net_height}')


    def tile_image(self, filepath):
        print(f'Source file: {filepath}')
#        if isinstance(filepath, Path):
#            filepath = str(filepath)

        self.img = Image.open(filepath)
        img_width, img_height = self.img.size
        print(f'image size: {img_width} x {img_height}')

        # define tile dimensions and number of tiles
        self.h_tiles = round(img_width / self.net_width)
        self.v_tiles = round(img_height / self.net_height)
        print(f'number of tiles: {self.h_tiles} x {self.v_tiles}')

        self.tile_width = img_width // self.h_tiles
        self.tile_height = img_height // self.v_tiles
        print(f'tile size: {self.tile_width} x {self.tile_height}')

        for ht in range(self.h_tiles):
            x1, x2 = self._get_side_coords(
                img_width, ht, self.tile_width)
            for vt in range(self.v_tiles):
                y1, y2 = self._get_side_coords(
                    img_height, vt, self.tile_height)
                box = (x1, y1, x2, y2)
                tile = self.img.crop(box)
                save_path = self.save_tile(tile, (ht, vt), filepath)
                print(f'Saved {save_path}')

    def tile_dir_images(self):
        print(f'Source directory: {self.srcdir}')
        files = os.listdir(self.srcdir)
        for f in files:
            pf = self.srcdir / f
            if pf.suffix in ('.jpg', '.png'):
                self.tile_image(pf)

    def _get_side_coords(self, img_side_size: int, tile_number: int,
                         tile_side_size: int):
        """Get coordinates of a tile in the image."""
        c1 = tile_number * tile_side_size
        c2 = (tile_number + 1) * tile_side_size
        if img_side_size - c1 < tile_side_size:
            c2 = img_side_size

        return c1, c2

    def save_tile(self, tile, tile_number, filepath):
        filename = Path(filepath).name
        f_list = str(filename).split('.')
        del f_list[-1]
        fname = self.destdir / '.'.join(f_list)
        tile_name = (f'{fname}.tile.{tile.size[0]}x{tile.size[1]}.'
                     f'{tile_number[0]}-{tile_number[1]}.jpg')
        save_path = tile_name
        tile.save(save_path)
        return save_path


if __name__ == '__main__':
    #filename = '2015-06-04_10_25_31.png'
    #p = Path('.')
    #image_dir = p / 'data'
    #infile = image_dir / filename
    net_size = 608, 608

    srcdir =  '/home/berno/Documents/img_tiling'
    destdir =  '/home/berno/Documents/img_tiling/tiled/'
    tiler = Tiler(net_size, srcdir, destdir)
    tiler.tile_dir_images()
#    tiler = Tiler(net_size)
#    tiler.tile_image(infile)