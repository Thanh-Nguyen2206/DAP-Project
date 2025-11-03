"""
Advanced Visualization Module for Stock Analysis Dashboard
Implements advanced chart types for comprehensive data visualization

Features:
- Waffle Charts for portfolio composition
- Word Clouds for sentiment analysis
- Area Plots for volume analysis
- Advanced statistical visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

# Optional imports with fallbacks
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from pywaffle import Waffle
    PYWAFFLE_AVAILABLE = True
except ImportError:
    PYWAFFLE_AVAILABLE = False

try:
    import folium
    import geopandas as gpd
    CHOROPLETH_AVAILABLE = True
except ImportError:
    CHOROPLETH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """Advanced visualization tools for stock market analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.logger.info("Advanced Visualizer initialized")
    
    def create_waffle_chart(self, data_dict, title="Portfolio Composition"):
        """
        Create a waffle chart for portfolio composition with fallback to pie chart
        """
        try:
            import matplotlib.pyplot as plt
            
            if PYWAFFLE_AVAILABLE:
                from pywaffle import Waffle
                
                # Convert percentages to counts (scale to 100 for waffle)
                total = sum(data_dict.values())
                waffle_data = {k: int(v) for k, v in data_dict.items()}
                
                # Create figure
                fig = plt.figure(
                    FigureClass=Waffle,
                    rows=10,
                    values=waffle_data,
                    colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"],
                    title={
                        'label': title,
                        'loc': 'center',
                        'fontdict': {'fontsize': 16, 'fontweight': 'bold'}
                    },
                    labels=["{0} ({1}%)".format(k, v) for k, v in data_dict.items()],
                    legend={
                        'loc': 'upper left',
                        'bbox_to_anchor': (1.1, 1)
                    },
                    figsize=(12, 8)
                )
            
            self.logger.info(f"Waffle chart created for {len(data_dict)} assets")
            return fig
            
        except Exception as e:
            self.logger.warning(f"Waffle chart failed, using pie chart fallback: {e}")
            # Fallback to pie chart
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
                
                wedges, texts, autotexts = ax.pie(
                    data_dict.values(), 
                    labels=data_dict.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors[:len(data_dict)]
                )
                
                ax.set_title(title, fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                self.logger.info(f"Pie chart fallback created for {len(data_dict)} assets")
                return fig
                
            except Exception as fallback_e:
                self.logger.error(f"Both waffle and pie chart failed: {fallback_e}")
                return None
    
    def create_wordcloud_visualization(self, word_data: Dict[str, float], 
                                     title: str = "Stock Performance Word Cloud") -> Any:
        """
        Create word cloud visualization from word frequency data
        
        Args:
            word_data: Dictionary with words as keys and frequencies as values
            title: Chart title
            
        Returns:
            Matplotlib figure object
        """
        try:
            if not WORDCLOUD_AVAILABLE:
                raise ImportError("WordCloud library not available")
                
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Generate word cloud from frequency dict
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(word_data)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            self.logger.info(f"Word cloud created from {len(word_data)} text entries")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating word cloud: {e}")
            return None
    
    def create_wordcloud_from_text(self, text_data: List[str], 
                                  title: str = "Stock News Sentiment") -> Any:
        """
        Create word cloud visualization for text analysis
        
        Args:
            text_data: List of text strings (news, comments, etc.)
            title: Chart title
            
        Returns:
            Matplotlib figure object
        """
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Combine all text
            combined_text = ' '.join(text_data)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate(combined_text)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            self.logger.info(f"Word cloud created from {len(text_data)} text entries")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating word cloud: {e}")
            return None
    
    def create_advanced_area_plot(self, df: pd.DataFrame, 
                                 x_col: str,
                                 y_cols: List[str],
                                 title: str = "Advanced Multi-layer Analysis") -> go.Figure:
        """
        Create advanced area plot for multi-layer analysis
        
        Args:
            df: DataFrame with data
            x_col: X-axis column name
            y_cols: List of Y-axis column names
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Validate input columns
            if x_col not in df.columns:
                available_cols = list(df.columns)
                self.logger.error(f"Column '{x_col}' not found. Available columns: {available_cols}")
                raise ValueError(f"Column '{x_col}' not found in DataFrame. Available columns: {available_cols}")
            
            # Check which y_cols exist
            valid_y_cols = [col for col in y_cols if col in df.columns]
            if not valid_y_cols:
                self.logger.error(f"None of the y_cols {y_cols} found in DataFrame columns: {list(df.columns)}")
                raise ValueError(f"None of the specified columns found: {y_cols}")
            
            fig = go.Figure()
            
            colors = ['rgba(74, 144, 226, 0.3)', 'rgba(255, 107, 107, 0.3)', 'rgba(46, 204, 113, 0.3)']
            
            for i, col in enumerate(valid_y_cols):
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    fill='tonexty' if i > 0 else 'tozeroy',
                    mode='lines',
                    name=col,
                    fillcolor=colors[i % len(colors)],
                    line=dict(width=1)
                ))
            
            fig.update_layout(
                title=f" {title}",
                xaxis_title=x_col,
                yaxis_title="Value",
                height=500,
                plot_bgcolor='white',
                showlegend=True
            )
            
            self.logger.info(f"Advanced area plot created with {len(valid_y_cols)} layers")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating area plot: {e}")
            return self._create_error_figure("Error creating area plot")
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                  title: str = "Correlation Analysis") -> go.Figure:
        """
        Create correlation heatmap for stock data
        
        Args:
            df: DataFrame with stock data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f" {title}",
                width=600,
                height=500,
                plot_bgcolor='white'
            )
            
            self.logger.info(f"Correlation heatmap created for {len(df.columns)} features")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            return self._create_error_figure("Error creating correlation heatmap")
    
    def create_distribution_analysis(self, data: pd.Series, 
                                   title: str = "Price Distribution Analysis") -> go.Figure:
        """
        Create distribution analysis plot
        
        Args:
            data: Series with price data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Distribution', 'Box Plot'),
                horizontal_spacing=0.1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=50,
                    name='Distribution',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=data,
                    name='Box Plot',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f" {title}",
                height=400,
                showlegend=False,
                plot_bgcolor='white'
            )
            
            self.logger.info(f"Distribution analysis created for {len(data)} data points")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating distribution analysis: {e}")
            return self._create_error_figure("Error creating distribution analysis")
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=f" {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    def create_choropleth_map(self, data_dict: Dict[str, float], 
                            title: str = "Global Stock Market Performance",
                            map_type: str = "world") -> Any:
        """
        Create choropleth map visualization for geographic data
        
        Args:
            data_dict: Dictionary with country codes/names as keys and values as data
            title: Map title
            map_type: Type of map ('world', 'usa', 'europe')
            
        Returns:
            Folium map object or Plotly figure as fallback
        """
        try:
            if not CHOROPLETH_AVAILABLE:
                # Fallback to plotly choropleth
                return self._create_plotly_choropleth(data_dict, title)
            
            import folium
            
            # Create base map with better zoom and positioning for larger display
            if map_type == "world":
                m = folium.Map(
                    location=[20, 0], 
                    zoom_start=2,
                    width='100%',
                    height='100%',
                    prefer_canvas=True  # Better performance for large maps
                )
                
                # Sample world data for demonstration - fixed country names to match GeoJSON
                world_data = {
                    'United States of America': data_dict.get('USA', 2.5),
                    'China': data_dict.get('China', 1.8),
                    'Japan': data_dict.get('Japan', 1.2),
                    'Germany': data_dict.get('Germany', 1.5),
                    'United Kingdom': data_dict.get('UK', 1.1),
                    'France': data_dict.get('France', 0.9),
                    'India': data_dict.get('India', 3.2),
                    'Canada': data_dict.get('Canada', 1.3),
                    'Australia': data_dict.get('Australia', 1.0),
                    'South Korea': data_dict.get('Korea', 2.1),
                    'Brazil': data_dict.get('Brazil', 1.6),
                    'Mexico': data_dict.get('Mexico', 1.4),
                    'Italy': data_dict.get('Italy', 0.8),
                    'Spain': data_dict.get('Spain', 0.7),
                    'Netherlands': data_dict.get('Netherlands', 1.9)
                }
                
                # Create data for folium choropleth (need list format)
                import pandas as pd
                df = pd.DataFrame(list(world_data.items()), columns=['Country', 'Performance'])
                
                # Add real choropleth layer - single layer only to avoid duplication
                try:
                    folium.Choropleth(
                        geo_data="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
                        name="Stock Market Performance",
                        data=df,
                        columns=['Country', 'Performance'],
                        key_on="feature.properties.name",
                        fill_color="YlOrRd",
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name=f"{title} (%)",
                        nan_fill_color="lightgray"
                    ).add_to(m)
                except Exception as choropleth_error:
                    # If choropleth fails, just add simple markers as backup
                    self.logger.warning(f"Choropleth failed: {choropleth_error}, using markers")
                    for country, performance in world_data.items():
                        country_coords = {
                            'United States of America': [39.8283, -98.5795], 'China': [35.8617, 104.1954], 
                            'Japan': [36.2048, 138.2529], 'Germany': [51.1657, 10.4515],
                            'United Kingdom': [55.3781, -3.4360], 'France': [46.6034, 1.8883],
                            'India': [20.5937, 78.9629], 'Canada': [56.1304, -106.3468],
                            'Australia': [-25.2744, 133.7751], 'South Korea': [35.9078, 127.7669],
                            'Brazil': [-14.2350, -51.9253], 'Mexico': [23.6345, -102.5528],
                            'Italy': [41.8719, 12.5674], 'Spain': [40.4637, -3.7492],
                            'Netherlands': [52.1326, 5.2913]
                        }
                        if country in country_coords:
                            folium.CircleMarker(
                                location=country_coords[country],
                                radius=8,
                                popup=f"{country}: {performance}%",
                                color='red' if performance >= 2.5 else 'orange' if performance >= 1.5 else 'yellow',
                                fillOpacity=0.6
                            ).add_to(m)
                
            elif map_type == "usa":
                m = folium.Map(location=[39.8, -98.5], zoom_start=4)
                # Add US states data here if needed
                
            # Add title (only once)
            title_html = f'''
                         <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
                         '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            self.logger.info(f"Choropleth map created: {title}")
            return m
            
        except Exception as e:
            self.logger.error(f"Error creating choropleth map: {e}")
            return self._create_plotly_choropleth(data_dict, title)
    
    def _create_plotly_choropleth(self, data_dict: Dict[str, float], title: str) -> Any:
        """Fallback choropleth using Plotly"""
        try:
            import plotly.graph_objects as go
            
            # Extended country codes and data to match all sliders
            countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'CAN', 'AUS', 'KOR', 'BRA', 'MEX', 'ITA', 'ESP', 'NLD']
            country_mapping = ['USA', 'China', 'Japan', 'Germany', 'UK', 'France', 'India', 'Canada', 'Australia', 'Korea', 'Brazil', 'Mexico', 'Italy', 'Spain', 'Netherlands']
            values = [data_dict.get(country, 1.0) for country in country_mapping]
            
            fig = go.Figure(data=go.Choropleth(
                locations=countries,
                z=values,
                locationmode='ISO-3',
                colorscale='Viridis',
                reversescale=True,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_title=f"{title}<br>Performance %",
            ))
            
            fig.update_layout(
                title_text=title,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular'
                ),
                height=1200,  # Increased height by 50% (800 → 1200)
                width=1800    # Increased width by 50% (1200 → 1800)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating plotly choropleth: {e}")
            return self._create_error_figure(f"Failed to create choropleth map: {str(e)}")

# Helper functions for demo data
def generate_sample_portfolio() -> Dict[str, float]:
    """Generate sample portfolio data for demo"""
    return {
        'AAPL': 25.5,
        'GOOGL': 20.3,
        'MSFT': 18.7,
        'TSLA': 15.2,
        'META': 12.1,
        'NVDA': 8.2
    }

def generate_sample_news() -> List[str]:
    """Generate sample news text for word cloud demo"""
    return [
        "Apple stock rises on strong iPhone sales growth technology innovation",
        "Microsoft cloud services showing excellent revenue growth Azure expansion",
        "Tesla electric vehicle production increases sustainability focus market leader",
        "Google advertising revenue beats expectations search dominance continues",
        "Meta metaverse investments showing progress virtual reality advancement",
        "NVIDIA AI chip demand surges artificial intelligence machine learning boom"
    ]

def generate_sample_geographic_data() -> Dict[str, float]:
    """Generate sample geographic market data for choropleth demo"""
    return {
        'USA': 2.5,
        'China': 1.8,
        'Japan': 1.2,
        'Germany': 1.5,
        'UK': 1.1,
        'France': 0.9,
        'India': 3.2,
        'Canada': 1.3,
        'Australia': 1.0,
        'Korea': 2.1,
        'Brazil': 2.8,
        'Mexico': 1.7,
        'Italy': 0.8,
        'Spain': 0.7,
        'Netherlands': 1.4
    }

if __name__ == "__main__":
    # Demo usage
    viz = AdvancedVisualizer()
    
    # Test portfolio waffle chart
    portfolio = generate_sample_portfolio()
    waffle_fig = viz.create_waffle_chart(portfolio)
    print(" Waffle chart demo completed")
    
    # Test word cloud
    news_data = generate_sample_news()
    wordcloud_fig = viz.create_wordcloud_visualization(news_data)
    print(" Word cloud demo completed")
    
    # Test choropleth map
    geo_data = generate_sample_geographic_data()
    choropleth_map = viz.create_choropleth_map(geo_data, "Global Market Performance Demo")
    print(" Choropleth map demo completed")
    
    print(" Advanced Visualization Module ready!")