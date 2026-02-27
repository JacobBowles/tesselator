import argparse
import pandas as pd
from vessel import TaperedVessel
from processor import PatternProcessor

def main():
    parser = argparse.ArgumentParser(description="Islamic Pattern Pottery Mapper")
    parser.add_argument("pattern", help="Path to the single-tile image")
    parser.add_argument("measurements", help="CSV with 'arc_length,circumference'")
    parser.add_argument("--tiles", type=int, default=8, help="Number of tiles to fit around the rim")
    parser.add_argument("--output", default="template.png", help="Output filename")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.measurements)
    c_top = df.iloc[0]['circumference']
    c_bottom = df.iloc[1]['circumference']
    slant = df.iloc[1]['arc_length'] - df.iloc[0]['arc_length']

    # Process
    print(f"--- Processing Vessel: {c_top}mm -> {c_bottom}mm over {slant}mm ---")
    vessel = TaperedVessel(c_top, c_bottom, slant)
    processor = PatternProcessor(args.pattern, args.tiles)
    
    result = processor.generate_template(vessel)
    result.save(args.output)
    print(f"Success! Template saved to {args.output}")

if __name__ == "__main__":
    main()