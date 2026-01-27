from tempocom.modules.streamlit import Page
import streamlit as st
import os

class Maintenance(Page):
    title = "TEMPOCOM - Maintenance"
    
    def render(self):
        assets_path = os.getenv('ASSETS_PATH')
        
        # Center content using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Logo
            if assets_path:
                st.image(f"{assets_path}/logo.png", width=150)
            
            # Main icon and title
            st.markdown("<div style='text-align: center; font-size: 120px; color: #ff6b6b;'>🔧</div>", unsafe_allow_html=True)
            
            st.markdown("<h1 style='text-align: center; font-size: 3em; color: #2c3e50; font-weight: bold;'>Under Maintenance</h1>", unsafe_allow_html=True)
            
            st.markdown("<p style='text-align: center; font-size: 1.4em; color: #7f8c8d; line-height: 1.6;'>TEMPOCOM is temporarily unavailable for technical improvements</p>", unsafe_allow_html=True)
            
            # Main message
            st.info("🚀 **We're working to bring you a better experience**\n\nOur team is currently performing important updates to improve performance and add new features to your railway Digital Twin.")
            
            # Maintenance details
            st.subheader("📋 Maintenance details:")
            st.markdown("""
            - **Optimization** of simulation algorithms
            - **Update** of network database  
            - **Enhancement** of user interface
            - **Strengthening** of data security
            """)
            
            # Contact info
            st.success("""
            **📞 Need help?**
            
            For any urgent matter or questions, contact the BRAIN team:
            
            **Email:** support@brain-research.be  
            **Phone:** +32 (0)2 XXX XX XX
            """)
            
            # Status info
            st.markdown("---")
            st.caption(f"⏱️ Estimated duration: Under evaluation")
            st.caption(f"🔄 Last update: {st.session_state.get('current_time', 'now')}")
            
            # Refresh button
            if st.button("🔄 Refresh page", type="primary", use_container_width=True):
                st.rerun()

if __name__ == "__main__":
    Maintenance()