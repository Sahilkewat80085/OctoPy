from Octopy.report import generate_report

print("[TEST] Running report generation test...")
generate_report("model.pkl", "X_test.csv", "y_test.csv", format="both")
print("[TEST] Test completed successfully!")