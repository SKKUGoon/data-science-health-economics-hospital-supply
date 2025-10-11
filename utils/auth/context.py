# """
# Hospital context utilities for page-level access
# """
# from typing import Optional
# from .hospital_profile import HospitalProfile, get_hospital_manager

# # Optional streamlit import for UI functionality
# try:
#     import streamlit as st
#     STREAMLIT_AVAILABLE = True
# except ImportError:
#     STREAMLIT_AVAILABLE = False


# def get_current_hospital_context() -> Optional[HospitalProfile]:
#     """
#     Get current hospital profile from session state

#     Returns:
#         Current hospital profile if logged in, None otherwise
#     """
#     manager = get_hospital_manager()
#     return manager.get_current_hospital()


# def is_user_admin() -> bool:
#     """
#     Check if current user has admin privileges

#     Returns:
#         True if user is admin, False otherwise
#     """
#     manager = get_hospital_manager()
#     if manager.is_logged_in():
#         current_hospital = manager.get_current_hospital()
#         return current_hospital.is_admin if current_hospital else False
#     return False


# def get_current_hospital_id() -> Optional[str]:
#     """
#     Get current hospital ID

#     Returns:
#         Hospital ID if logged in, None otherwise
#     """
#     manager = get_hospital_manager()
#     if manager.is_logged_in():
#         current_hospital = manager.get_current_hospital()
#         return current_hospital.hospital_id if current_hospital else None
#     return None


# def require_hospital_context() -> HospitalProfile:
#     """
#     Require hospital context - shows error and stops execution if not available

#     Returns:
#         Current hospital profile

#     Raises:
#         RuntimeError if no hospital context available (or SystemExit if streamlit available)
#     """
#     manager = get_hospital_manager()
#     if not manager.is_logged_in():
#         error_msg = "로그인이 필요합니다. Home 페이지에서 로그인해주세요."
#         if STREAMLIT_AVAILABLE:
#             st.error(error_msg)
#             st.stop()
#         else:
#             raise RuntimeError("Hospital login required")

#     current_hospital = manager.get_current_hospital()
#     if not current_hospital:
#         error_msg = "병원 정보를 찾을 수 없습니다. Home 페이지에서 로그인해주세요."
#         if STREAMLIT_AVAILABLE:
#             st.error(error_msg)
#             st.stop()
#         else:
#             raise RuntimeError("Hospital profile not found")
#     return current_hospital


# def display_hospital_header():
#     """Display current hospital information in header (requires streamlit)"""
#     if not STREAMLIT_AVAILABLE:
#         return

#     manager = get_hospital_manager()
#     if manager.is_logged_in():
#         current_hospital = manager.get_current_hospital()
#         if current_hospital:
#             st.sidebar.info(f"🏥 **현재 병원:** {current_hospital.hospital_name}")
#             st.sidebar.info(f"🆔 **병원 ID:** {current_hospital.hospital_id}")
#             if current_hospital.is_admin:
#                 st.sidebar.success("👑 **관리자 권한**")
#         else:
#             st.sidebar.warning("⚠️ 병원 정보 없음")
#     else:
#         st.sidebar.warning("⚠️ 로그인 필요")


# def get_hospital_context_summary() -> Optional[dict]:
#     """
#     Get hospital context summary as dictionary (non-UI version).

#     Returns:
#         Dictionary with hospital context information or None if not logged in
#     """
#     manager = get_hospital_manager()
#     if not manager.is_logged_in():
#         return None

#     current_hospital = manager.get_current_hospital()
#     if not current_hospital:
#         return None

#     return {
#         "hospital_id": current_hospital.hospital_id,
#         "hospital_name": current_hospital.hospital_name,
#         "is_admin": current_hospital.is_admin,
#         "is_valid": current_hospital.is_valid_for_operations()
#     }