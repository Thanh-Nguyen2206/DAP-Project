"""
Simplified Explainable AI Module for Stock Prediction Models
Provides basic model interpretability features with graceful fallbacks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

# XAI Libraries with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplainableAI:
    """Simplified Explainable AI tools for stock prediction models"""
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = []
        self.model = None
        logger.info("Explainable AI initialized")
        
    def setup_explainers(self, model: Any, X_train: pd.DataFrame) -> bool:
        """Setup explainers for the given model"""
        try:
            self.model = model
            self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
            
            # Setup SHAP if available
            if SHAP_AVAILABLE and hasattr(model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(model)
                logger.info("SHAP explainer setup completed")
                
            # Setup LIME if available
            if LIME_AVAILABLE:
                self.lime_explainer = LimeTabularExplainer(
                    X_train.values,
                    feature_names=self.feature_names,
                    mode='regression'
                )
                logger.info("LIME explainer setup completed")
                
            return True
        except Exception as e:
            logger.error(f"Error setting up explainers: {e}")
            return False
    
    def get_shap_feature_importance(self, X_test: pd.DataFrame, title: str = "SHAP Feature Importance") -> Optional[plt.Figure]:
        """Generate SHAP feature importance plot"""
        try:
            if not SHAP_AVAILABLE or self.shap_explainer is None:
                return self.create_fallback_explanation(X_test, title)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_test.values)
            
            # Create SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
            plt.title(title)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error generating SHAP feature importance: {e}")
            return self.create_fallback_explanation(X_test, title)
    
    def get_shap_waterfall(self, X_test: pd.DataFrame, instance_idx: int = 0, title: str = "SHAP Waterfall") -> Optional[plt.Figure]:
        """Generate SHAP waterfall plot for a specific instance"""
        try:
            if not SHAP_AVAILABLE or self.shap_explainer is None:
                return self.create_fallback_explanation(X_test, title)
            
            # Calculate SHAP values for the specific instance
            shap_values = self.shap_explainer.shap_values(X_test.iloc[instance_idx:instance_idx+1].values)
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.shap_explainer.expected_value,
                    data=X_test.iloc[instance_idx].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title(title)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error generating SHAP waterfall: {e}")
            return self.create_fallback_explanation(X_test, title)
    
    def get_lime_explanation(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, instance_idx: int = 0) -> Optional[plt.Figure]:
        """Generate LIME explanation for a specific instance"""
        try:
            if not LIME_AVAILABLE or self.lime_explainer is None:
                return self.create_fallback_explanation(X_test, "LIME Explanation")
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_test.iloc[instance_idx].values,
                model.predict,
                num_features=min(10, len(self.feature_names))
            )
            
            # Create LIME plot
            fig, ax = plt.subplots(figsize=(10, 8))
            explanation.as_pyplot_figure()
            plt.title("LIME Local Explanation")
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return self.create_fallback_explanation(X_test, "LIME Explanation")
    
    def create_fallback_explanation(self, features_df: pd.DataFrame, title: str = "Model Analysis") -> plt.Figure:
        """Create a fallback explanation when SHAP/LIME are not available"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{title} - Basic Analysis', fontsize=16, fontweight='bold')
            
            # 1. Feature Correlation
            if len(features_df.columns) > 1:
                corr_matrix = features_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1, fmt='.2f')
                ax1.set_title('Feature Correlation Matrix')
            else:
                ax1.text(0.5, 0.5, 'Need multiple features\nfor correlation analysis', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Feature Correlation')
            
            # 2. Feature Importance (if model available)
            if self.model and hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = features_df.columns
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                top_indices = indices[:min(10, len(indices))]
                
                ax2.barh(range(len(top_indices)), importances[top_indices])
                ax2.set_title('Feature Importance (Random Forest)')
                ax2.set_xlabel('Importance')
                ax2.set_yticks(range(len(top_indices)))
                ax2.set_yticklabels([feature_names[i] for i in top_indices])
            else:
                ax2.text(0.5, 0.5, 'Feature importance\nnot available for this model', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Feature Importance')
            
            # 3. Feature Statistics
            if len(features_df.columns) > 0:
                stats_data = {
                    'Mean': features_df.mean(),
                    'Std': features_df.std(),
                    'Min': features_df.min(),
                    'Max': features_df.max()
                }
                stats_df = pd.DataFrame(stats_data)
                
                # Plot feature statistics
                stats_df.plot(kind='bar', ax=ax3)
                ax3.set_title('Feature Statistics')
                ax3.set_xlabel('Features')
                ax3.set_ylabel('Values')
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend()
            
            # 4. Sample Data Distribution
            if len(features_df.columns) > 0:
                # Plot distribution of the first feature
                feature_to_plot = features_df.columns[0]
                ax4.hist(features_df[feature_to_plot], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax4.axvline(features_df[feature_to_plot].mean(), color='red', linestyle='--', 
                           label=f'Mean: {features_df[feature_to_plot].mean():.2f}')
                ax4.set_title(f'Distribution of {feature_to_plot}')
                ax4.set_xlabel(feature_to_plot)
                ax4.set_ylabel('Frequency')
                ax4.legend()
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating fallback explanation: {e}")
            # Return a simple error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error generating explanation:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Analysis Error')
            return fig
    
    def compare_explanations(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           title: str = "Model Comparison") -> plt.Figure:
        """Compare different explanation methods"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # 1. Model Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_test.columns
                indices = np.argsort(importances)[::-1][:10]
                
                ax1.barh(range(len(indices)), importances[indices])
                ax1.set_title('Random Forest Feature Importance')
                ax1.set_yticks(range(len(indices)))
                ax1.set_yticklabels([feature_names[i] for i in indices])
            
            # 2. Feature Correlation with Target
            # This would need the target variable, so we'll show feature correlations
            if len(X_test.columns) > 1:
                corr_with_first = X_test.corr().iloc[0, 1:].abs().sort_values(ascending=True)
                ax2.barh(range(len(corr_with_first)), corr_with_first.values)
                ax2.set_title(f'Correlation with {X_test.columns[0]}')
                ax2.set_yticks(range(len(corr_with_first)))
                ax2.set_yticklabels(corr_with_first.index)
            
            # 3. Prediction Confidence (sample)
            if len(X_test) >= 10:
                sample_indices = range(min(10, len(X_test)))
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_test.iloc[:10])
                    ax3.plot(sample_indices, predictions, 'o-', label='Predictions')
                    ax3.set_title('Sample Predictions')
                    ax3.set_xlabel('Sample Index')
                    ax3.set_ylabel('Predicted Value')
                    ax3.legend()
            
            # 4. Model Performance Metrics (simplified)
            ax4.text(0.1, 0.8, 'Model Information:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.1, 0.7, f'Model Type: {type(model).__name__}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.6, f'Features: {len(X_test.columns)}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.5, f'Samples: {len(X_test)}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.4, f'SHAP Available: {SHAP_AVAILABLE}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.3, f'LIME Available: {LIME_AVAILABLE}', fontsize=12, transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Model Summary')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison: {e}")
            return self.create_fallback_explanation(X_test, title)