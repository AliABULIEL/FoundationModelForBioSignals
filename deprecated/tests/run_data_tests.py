#!/usr/bin/env python
"""
Test runner for data.py SPEC compliance.
Executes all tests and generates a summary report.
"""

import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime


def run_tests():
    """Run all data.py tests and generate report."""
    
    print("=" * 60)
    print("VitalDB Data.py SPEC Compliance Test Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test files
    test_files = [
        "tests/test_data_spec_compliance.py",
        "tests/test_data_edge_cases.py"
    ]
    
    results = {}
    
    # Run each test file
    for test_file in test_files:
        print(f"\n📋 Running: {test_file}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Run pytest with verbose output
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--no-header"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
            # Parse output for pass/fail counts
            output = result.stdout
            
            # Look for pytest summary
            if "passed" in output:
                # Extract test counts
                lines = output.split('\n')
                for line in lines:
                    if 'passed' in line or 'failed' in line:
                        results[test_file] = {
                            'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                            'output': line.strip(),
                            'time': f"{elapsed:.2f}s"
                        }
                        print(f"  {line.strip()}")
                        break
            else:
                results[test_file] = {
                    'status': 'ERROR',
                    'output': 'No test results found',
                    'time': f"{elapsed:.2f}s"
                }
                
        except subprocess.TimeoutExpired:
            results[test_file] = {
                'status': 'TIMEOUT',
                'output': 'Test execution timed out',
                'time': '60.00s'
            }
        except Exception as e:
            results[test_file] = {
                'status': 'ERROR',
                'output': str(e),
                'time': '0.00s'
            }
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    for test_file, result in results.items():
        status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"\n{status_symbol} {Path(test_file).name}")
        print(f"   Status: {result['status']}")
        print(f"   Time: {result['time']}")
        print(f"   Result: {result['output']}")
    
    # Check SPEC compliance
    print("\n" + "=" * 60)
    print("🎯 SPEC COMPLIANCE CHECK")
    print("=" * 60)
    
    spec_items = [
        ("QC: Flatline Detection", "test_flatline_detection"),
        ("QC: Spike Detection", "test_spike_detection"),
        ("QC: Physiologic Bounds", "test_physiologic_bounds"),
        ("QC: PPG SQI", "test_ppg_sqi"),
        ("Filter: PPG Chebyshev", "test_ppg_filter_spec"),
        ("Filter: ECG Butterworth", "test_ecg_filter_spec"),
        ("Filter: ABP Butterworth", "test_abp_filter_spec"),
        ("Filter: EEG Wavelet", "test_eeg_filter_spec"),
        ("Sampling: Rate Conversion", "test_target_sampling_rates"),
        ("Window: 10s/5s hop", "test_window_parameters"),
        ("Clinical: Data Extraction", "test_clinical_fields_extraction"),
        ("Cache: Versioning", "test_cache_versioning"),
        ("Splits: Patient-level", "test_patient_level_splits"),
    ]
    
    print("\nKey SPEC Requirements:")
    for spec_name, test_name in spec_items:
        # Check if test exists in results
        found = any(test_name in str(r.get('output', '')) for r in results.values())
        status = "✅" if found else "⚠️"
        print(f"  {status} {spec_name}")
    
    # Run coverage if available
    print("\n" + "=" * 60)
    print("📈 CODE COVERAGE")
    print("=" * 60)
    
    try:
        coverage_cmd = [
            sys.executable, "-m", "pytest",
            "--cov=data",
            "--cov-report=term-missing:skip-covered",
            "--no-header",
            "-q",
            "tests/test_data_spec_compliance.py",
            "tests/test_data_edge_cases.py"
        ]
        
        coverage_result = subprocess.run(
            coverage_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if coverage_result.returncode == 0:
            # Extract coverage percentage
            for line in coverage_result.stdout.split('\n'):
                if 'data.py' in line or 'TOTAL' in line:
                    print(f"  {line.strip()}")
        else:
            print("  Coverage analysis not available (install pytest-cov)")
            
    except:
        print("  Coverage analysis skipped")
    
    # Final recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
1. Run tests before any data.py modifications:
   pytest tests/test_data_spec_compliance.py -v

2. Check edge cases for production:
   pytest tests/test_data_edge_cases.py -v

3. Generate HTML coverage report:
   pytest tests/test_data_*.py --cov=data --cov-report=html

4. Run specific test categories:
   pytest -k "test_quality_control" -v
   pytest -k "test_clinical" -v
   
5. For continuous integration:
   python tests/run_data_tests.py
    """)
    
    print("\n" + "=" * 60)
    print("Test suite execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("data.py").exists():
        print("Error: Please run from the repository root directory")
        print("Usage: python tests/run_data_tests.py")
        sys.exit(1)
    
    run_tests()
