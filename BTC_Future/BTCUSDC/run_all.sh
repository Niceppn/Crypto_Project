#!/bin/bash

# ==========================================
# Run All Models Training & Comparison
# ==========================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🤖 Maker Strategy Models - Complete Training Pipeline      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# ==========================================
# Step 0: Test Feature Engineering
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 STEP 0: Testing Feature Engineering"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 feature_engineering.py

if [ $? -ne 0 ]; then
    echo "❌ Feature engineering test failed!"
    exit 1
fi

echo ""
echo "✅ Feature engineering test passed!"
echo ""
read -p "Press ENTER to continue to Model 1 training..."
echo ""

# ==========================================
# Step 1: Train Model 1 (Multi-Class)
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔷 STEP 1: Training Model 1 (Multi-Class Classification)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 model_1_multiclass.py

if [ $? -ne 0 ]; then
    echo "❌ Model 1 training failed!"
    exit 1
fi

echo ""
echo "✅ Model 1 training complete!"
echo ""
read -p "Press ENTER to continue to Model 2 training..."
echo ""

# ==========================================
# Step 2: Train Model 2 (Dual Regression)
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔶 STEP 2: Training Model 2 (Dual Regression)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 model_2_dual_regression.py

if [ $? -ne 0 ]; then
    echo "❌ Model 2 training failed!"
    exit 1
fi

echo ""
echo "✅ Model 2 training complete!"
echo ""
read -p "Press ENTER to continue to Model 3 setup..."
echo ""

# ==========================================
# Step 3: Create Model 3 (Hybrid)
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💎 STEP 3: Creating Model 3 (Hybrid Approach)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 model_3_hybrid.py

if [ $? -ne 0 ]; then
    echo "❌ Model 3 setup failed!"
    exit 1
fi

echo ""
echo "✅ Model 3 setup complete!"
echo ""
read -p "Press ENTER to run backtest comparison..."
echo ""

# ==========================================
# Step 4: Run Backtest Comparison
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏁 STEP 4: Running Backtest Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 backtest_comparison.py

if [ $? -ne 0 ]; then
    echo "❌ Backtest comparison failed!"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 ALL STEPS COMPLETED SUCCESSFULLY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Generated files:"
echo "   ├── model_multiclass.txt"
echo "   ├── model_multiclass_metadata.json"
echo "   ├── model_buy_regressor.txt"
echo "   ├── model_sell_regressor.txt"
echo "   ├── model_dual_regression_metadata.json"
echo "   ├── hybrid_strategy.pkl"
echo "   ├── model_hybrid_config.json"
echo "   └── backtest_comparison.csv"
echo ""
echo "📊 Check backtest_comparison.csv for results!"
echo ""
