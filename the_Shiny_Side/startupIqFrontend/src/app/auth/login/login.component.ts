import { Component } from '@angular/core';
import { Auth, signInWithPopup, GoogleAuthProvider } from '@angular/fire/auth';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { Firestore, doc, getDoc, setDoc } from '@angular/fire/firestore';

@Component({
  selector: 'app-login',
  standalone: false,
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
})
export class LoginComponent {
  loginForm: FormGroup;

  constructor(
    private fb: FormBuilder,
    private router: Router,
    private auth: Auth,
    private firestore: Firestore
  ) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', Validators.required],
    });
  }

  onLogin() {
    if (this.loginForm.valid) {
      console.log(this.loginForm.value); // 👈 capture values
      this.router.navigate(['/dashboard']);
    } else {
      this.loginForm.markAllAsTouched(); // show errors if invalid
    }
  }

  async loginWithGoogle() {
    try {
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(this.auth, provider);
      const accessToken = await result.user?.getIdToken();
      const uid = result.user?.uid;

      localStorage.setItem('accessToken', accessToken || '');
      localStorage.setItem('uid', uid || '');

      const userRef = doc(this.firestore, 'user-data', uid!);
      const userSnap = await getDoc(userRef);

      if (!userSnap.exists()) {
        await setDoc(userRef, {
          email: result.user.email,
          name: result.user.displayName,
          createdAt: new Date(),
        });
        console.log('New user document created in Firestore ✅');
      } else {
        console.log('User document already exists ✅');
      }
      this.router.navigate(['/dashboard']);
    } catch (error) {
      console.error('Error logging in with Google:', error);
      throw error;
    }
  }

  // Easy getters for template
  get email() {
    return this.loginForm.get('email');
  }
  get password() {
    return this.loginForm.get('password');
  }
}
