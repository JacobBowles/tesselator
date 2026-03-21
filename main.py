import argparse
import pandas as pd
import sys
import os
from vessel import TaperedVessel
from processor import PatternProcessor


def validate_csv(df):
    required = {'arc_length', 'circumference'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    if len(df) < 2:
        raise ValueError("Vessel requires at least two measurement points.")


def main():
    parser = argparse.ArgumentParser(
        description="Pottery Pattern Mapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pattern",      help="Source pattern image (single tile)")
    parser.add_argument("measurements", help="CSV with arc_length, circumference")
    parser.add_argument("--tiles", type=int, default=10,
                        help="Number of horizontal tiles around the rim")
    parser.add_argument("--dpi",   type=int, default=300,
                        help="Output resolution in DPI")
    parser.add_argument("--slice", action="store_true",
                        help="Output a single flat tile slice (debug mode)")

    args = parser.parse_args()

    if not os.path.exists(args.pattern):
        print(f"Error: Pattern image '{args.pattern}' not found.")
        sys.exit(1)
    if not os.path.exists(args.measurements):
        print(f"Error: Measurement file '{args.measurements}' not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(args.measurements)
        validate_csv(df)

        pattern_base = os.path.splitext(os.path.basename(args.pattern))[0]
        vessel_base  = os.path.splitext(os.path.basename(args.measurements))[0]
        base_filename = f"{pattern_base}_{vessel_base}_{args.tiles}tiles"

        vessel    = TaperedVessel(df)
        processor = PatternProcessor(args.pattern, args.tiles)

        print(f"--- Processing Vessel ---")
        print(f"Top: {vessel.c_top}mm | Bottom: {vessel.c_bottom}mm | "
              f"Height: {vessel.s_total}mm")
        print(f"Tiles: {args.tiles}")

        if args.slice:
            print("Rendering single slice (debug mode)...")
            result_img = processor.render_slice(vessel, dpi=args.dpi)
            png_out = f"{base_filename}_slice.png"
            result_img.save(png_out, dpi=(args.dpi, args.dpi))
            print(f"Slice saved: {png_out}")
        else:
            print("Rendering full fan...")
            result_img = processor.render(vessel, dpi=args.dpi)
            png_out = f"{base_filename}.png"
            pdf_out = f"{base_filename}.pdf"
            result_img.save(png_out, dpi=(args.dpi, args.dpi))
            result_img.convert("RGB").save(pdf_out, resolution=args.dpi,
                                           save_all=True)
            print(f"--- Success! ---")
            print(f"Image: {png_out}")
            print(f"PDF:   {pdf_out}")

    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()