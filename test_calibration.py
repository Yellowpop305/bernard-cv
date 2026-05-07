"""
Calibration test — runs the CV pipeline against the 8 known calibration deals
and reports accuracy of tube length estimation.

Usage:
    python test_calibration.py http://localhost:8000
    python test_calibration.py https://bernard-cv.up.railway.app
"""

import sys
import requests

# 8 calibration deals with their actual tube lengths and design URLs
CALIBRATION_DEALS = [
    {
        'name': 'Donut character',
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/995153/189471_dummy.png',  # update with real URL
        'sign_width_cm': 109,
        'actual_tube_m': 17.31,
    },
    {
        'name': 'Culler Enterprises',
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/dummy/culler.png',
        'sign_width_cm': 48,
        'actual_tube_m': 5.23,
    },
    {
        'name': 'Ongles & Cie',
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/995367/189477_0b06b33772119e0a5eb69becb487b32e.jpg',
        'sign_width_cm': 68,
        'actual_tube_m': 7.72,
    },
    {
        'name': 'DJ Dimmy',
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/dummy/dj.png',
        'sign_width_cm': 91,
        'actual_tube_m': 11.10,
    },
    {
        'name': 'Leavenworth',
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/dummy/leavenworth.png',
        'sign_width_cm': 100,
        'actual_tube_m': 5.74,
    },
    {
        'name': 'Première Bulle',
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/dummy/premiere.png',
        'sign_width_cm': 147,
        'actual_tube_m': 15.03,
    },
    {
        'name': "L'amour Lé Dou",
        'image_url': 'https://yellowpop.nyc3.digitaloceanspaces.com/dummy/lamour.png',
        'sign_width_cm': 77,
        'actual_tube_m': 7.35,
    },
]


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8000'
    base_url = base_url.rstrip('/')

    print(f"\nTesting CV pipeline at: {base_url}\n")
    print(f"{'Deal':<25} {'Sign(cm)':<10} {'Actual(m)':<12} {'Predicted(m)':<14} {'Error':<10}")
    print('-' * 80)

    errors = []
    for deal in CALIBRATION_DEALS:
        try:
            res = requests.post(
                f"{base_url}/trace-tube",
                json={
                    'image_url': deal['image_url'],
                    'sign_width_cm': deal['sign_width_cm'],
                },
                timeout=30,
            )
            data = res.json()
            if 'error' in data:
                print(f"{deal['name']:<25} {deal['sign_width_cm']:<10} {deal['actual_tube_m']:<12} ERROR: {data['error']}")
                continue

            predicted = data['tube_length_m']
            actual = deal['actual_tube_m']
            error_pct = abs(predicted - actual) / actual * 100
            errors.append(error_pct)

            print(f"{deal['name']:<25} {deal['sign_width_cm']:<10} {actual:<12} {predicted:<14} ±{error_pct:.1f}%")
        except Exception as e:
            print(f"{deal['name']:<25} ERROR: {e}")

    if errors:
        print('-' * 80)
        print(f"Average error: ±{sum(errors)/len(errors):.1f}%")
        print(f"Max error: ±{max(errors):.1f}%")


if __name__ == '__main__':
    main()
