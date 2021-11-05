# Renames all the files in the cwd to the comic book naming convention

import argparse
import os
import sys

from slugify import slugify



parser = argparse.ArgumentParser('A utility that renames all files in the current working directory according to the chosen naming convention')
parser.add_argument('--dry-run', action='store_true', default=False, help='do not perform any actual rename operations')
parser.add_argument("name", type=str, nargs=1, help='name of the comic book')
args = parser.parse_args()

print(args.name)

for i, filename in enumerate(sorted(f for f in os.listdir('.') if f.lower().endswith('.jpg') or f.lower().endswith('.png')), 1):
    new_name = slugify(args.name[0] + ' ' + str(i)) + os.path.splitext(filename)[1].lower()
    print(filename, '->', new_name)

    if not args.dry_run:
        os.rename(filename, new_name)

