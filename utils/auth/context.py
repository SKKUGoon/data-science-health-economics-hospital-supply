"""
Hospital context utilities for page-level access
"""
import streamlit as st
from typing import Optional
from .hospital_profile import HospitalProfile, get_hospital_manager


def get_current_hospital_context() -> Optional[HospitalProfile]:
    """
    Get current hospital profile from session state

    Returns:
        Current hospital profile if logged in, None otherwise
    """
    manager = get_hospital_manager()
    return manager.get_current_hospital()


def is_user_admin() -> bool:
    """
    Check if current user has admin privileges

    Returns:
        True if user is admin, False otherwise
    """
    manager = get_hospital_manager()
    if manager.is_logged_in():
        current_hospital = manager.get_current_hospital()
        return current_hospital.is_admin if current_hospital else False
    return False


def get_current_hospital_id() -> Optional[str]:
    """
    Get current hospital ID

    Returns:
        Hospital ID if logged in, None otherwise
    """
    manager = get_hospital_manager()
    if manager.is_logged_in():
        current_hospital = manager.get_current_hospital()
        return current_hospital.hospital_id if current_hospital else None
    return None


def require_hospital_context() -> HospitalProfile:
    """
    Require hospital context - shows error and stops execution if not available

    Returns:
        Current hospital profile

    Raises:
        SystemExit if no hospital context available
    """
    manager = get_hospital_manager()
    if not manager.is_logged_in():
        st.error("로그인이 필요합니다. Home 페이지에서 로그인해주세요.")
        st.stop()

    current_hospital = manager.get_current_hospital()
    if not current_hospital:
        st.error("병원 정보를 찾을 수 없습니다. Home 페이지에서 로그인해주세요.")
        st.stop()
    return current_hospital


def display_hospital_header():
    """Display current hospital information in header"""
    manager = get_hospital_manager()
    if manager.is_logged_in():
        current_hospital = manager.get_current_hospital()
        if current_hospital:
            st.sidebar.info(f"🏥 **현재 병원:** {current_hospital.hospital_name}")
            st.sidebar.info(f"🆔 **병원 ID:** {current_hospital.hospital_id}")
            if current_hospital.is_admin:
                st.sidebar.success("👑 **관리자 권한**")
        else:
            st.sidebar.warning("⚠️ 병원 정보 없음")
    else:
        st.sidebar.warning("⚠️ 로그인 필요")