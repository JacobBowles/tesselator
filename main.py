import argparse
import pandas as pd
import sys
import os
from vessel import TaperedVessel
from processor import PatternProcessor

def validate_csv(df):
    """Ensure the CSV has the required columns and at least two points."""
    required = {'arc_length', 'circumference'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    if len(df) < 2:
        raise ValueError("Vessel requires at least two measurement points (top and bottom).")

def main():
    parser = argparse.ArgumentParser(
        description="Coding Partner's Pottery Pattern Mapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument("pattern", help="Path to the source pattern image (single tile)")
    parser.add_argument("measurements", help="Path to CSV file with vessel measurements")
    
    # Optional Arguments
    parser.add_argument("--tiles", type=int, default=10, help="Number of horizontal tiles around the rim")
    parser.add_argument("--output", default="unwrapped_guide.png", help="Name of the output image file")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the printable output")

    args = parser.parse_args()

    # 1. File Check
    if not os.path.exists(args.pattern):
        print(f"Error: Pattern image '{args.pattern}' not found.")
        sys.exit(1)
    if not os.path.exists(args.measurements):
        print(f"Error: Measurement file '{args.measurements}' not found.")
        sys.exit(1)

    try:
        # 2. Load and Validate Measurements
        df = pd.read_csv(args.measurements)
        validate_csv(df)
        
        # We take the first and last points for the trapezoid calculation
        c_top = df.iloc[0]['circumference']
        c_bottom = df.iloc[-1]['circumference']
        slant = df.iloc[-1]['arc_length'] - df.iloc[0]['arc_length']

        # 3. Initialize Engines
        vessel = TaperedVessel(c_top, c_bottom, slant)
        processor = PatternProcessor(args.pattern, args.tiles)

        # 4. Execute Rendering
        print(f"--- Processing Vessel ---")
        print(f"Rim: {c_top}mm | Base: {c_bottom}mm | Height: {slant}mm")
        print(f"Tiling: {args.tiles} horizontal tiles (proportional vertical height)")
        
        result_img = processor.render(vessel, dpi=args.dpi)
        
        # 5. Save Output
        result_img.save(args.output, dpi=(args.dpi, args.dpi))
        print(f"--- Success! ---")
        print(f"Guide saved as: {args.output}")
        
        # Save as PDF for precise printing
        pdf_output = args.output.replace('.png', '.pdf')
        if not pdf_output.endswith('.pdf'):
            pdf_output += '.pdf'
            
        # Convert to RGB for PDF compatibility (removes transparency)
        pdf_img = result_img.convert("RGB")
        pdf_img.save(pdf_output, resolution=args.dpi, save_all=True)
        
        print(f"--- Success! ---")
        print(f"Image: {args.output}")
        print(f"Print-ready PDF: {pdf_output}")
        
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()