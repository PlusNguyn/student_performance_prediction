from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .forms import AccountUpdateForm, ProfileUpdateForm
from .models import UserProfile


class ProfileUpdateFormTests(TestCase):
    def test_student_cannot_change_assigned_student_id(self):
        user = User.objects.create_user(
            username="student1",
            password="secret-pass",
            first_name="Student",
            email="student@example.com",
        )
        UserProfile.objects.create(
            user=user,
            role=UserProfile.STUDENT,
            account_code="900001",
            phone="0900000001",
        )

        form = ProfileUpdateForm(
            data={
                "first_name": "Student",
                "last_name": "Updated",
                "email": "student@example.com",
                "role": UserProfile.LECTURER,
                "account_code": "999999",
                "phone": "0900000002",
                "gender": "female",
                "date_of_birth": "",
                "avatar": "",
                "is_active": "",
            },
            user_instance=user,
        )

        self.assertTrue(form.is_valid(), form.errors)
        form.save()

        user.refresh_from_db()
        profile = user.profile
        self.assertEqual(profile.role, UserProfile.STUDENT)
        self.assertEqual(profile.account_code, "900001")
        self.assertEqual(profile.phone, "0900000002")
        self.assertTrue(user.is_active)

    def test_admin_account_update_can_change_student_id(self):
        user = User.objects.create_user(
            username="student2",
            password="secret-pass",
            first_name="Student",
            email="student2@example.com",
        )
        UserProfile.objects.create(
            user=user,
            role=UserProfile.STUDENT,
            account_code="900002",
        )

        form = AccountUpdateForm(
            data={
                "first_name": "Student",
                "last_name": "",
                "email": "student2@example.com",
                "role": UserProfile.STUDENT,
                "account_code": "900222",
                "phone": "",
                "gender": "",
                "date_of_birth": "",
                "avatar": "",
                "is_active": "on",
            },
            user_instance=user,
        )

        self.assertTrue(form.is_valid(), form.errors)
        form.save()

        user.refresh_from_db()
        self.assertEqual(user.profile.account_code, "900222")


class RoleAccessTests(TestCase):
    def test_student_is_redirected_from_lecturer_dashboard(self):
        user = User.objects.create_user(username="student3", password="secret-pass")
        UserProfile.objects.create(user=user, role=UserProfile.STUDENT, account_code="900003")
        self.client.force_login(user)

        response = self.client.get(reverse("accounts:lecturer_dashboard"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("accounts:student_dashboard"))
