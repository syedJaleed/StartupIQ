import { Component } from '@angular/core';
import { Firestore, doc, updateDoc, getDoc, arrayUnion, onSnapshot  } from '@angular/fire/firestore';
import { Auth, onAuthStateChanged, User } from '@angular/fire/auth';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, debounceTime } from 'rxjs';


@Component({
  selector: 'app-main-dash',
  standalone: false,
  templateUrl: './main-dash.component.html',
  styleUrl: './main-dash.component.css',
})
export class MainDashComponent {
  projectForm!: FormGroup;
  projects: any[] = [];
  filteredProjects: any[] = [];
  unsubscribe: (() => void) | null = null;
  private searchSubject = new Subject<string>(); 
  projectDetails = {
    companyInfo: {
      name: '',
      industry: '',
      fundingStage: '',
      tagline: '',
    },
  };

  constructor(private firestore: Firestore, private auth: Auth, private fb: FormBuilder, private router: Router){}

  ngOnInit() {
    onAuthStateChanged(this.auth, (user: User | null) => {
      if (user) {
        console.log('User is logged in:', user.uid);
        this.getProjects(user.uid);
      } else {
        console.error('User not logged in');
        this.projects = [];
        this.filteredProjects = [];
      }
    });
    this.projectForm = this.fb.group({
      name: ['', Validators.required],
      industry: ['', Validators.required],
      fundingStage: ['', Validators.required],
      tagline: ['', Validators.required],
    });
     this.searchSubject.pipe(debounceTime(200)).subscribe((term) => {
      this.applySearch(term);
    });
  }
  
  async getProjects(uid: string) {
    const userRef = doc(this.firestore, 'user-data', uid);

    // ✅ Real-time listener
    this.unsubscribe = onSnapshot(userRef, (docSnap) => {
      if (docSnap.exists()) {
        const data = docSnap.data();
        this.projects = data['projects'] || [];
        this.filteredProjects = [...this.projects];
        console.log('Realtime projects update:', this.projects);
      }
    });
  }
  onSearch(term: string) {
    this.searchSubject.next(term);
  }
   private applySearch(term: string) {
    if (!term.trim()) {
      this.filteredProjects = [...this.projects];
      return;
    }

    const lower = term.toLowerCase();
    this.filteredProjects = this.projects.filter((proj) => {
      const name = proj.companyInfo?.name?.toLowerCase() || '';
      const industry = proj.companyInfo?.industry?.toLowerCase() || '';
      return name.includes(lower) || industry.includes(lower);
    });
  }

  async saveDetails() {
     if (this.projectForm.invalid) {
      this.projectForm.markAllAsTouched();
      return;
    }

    console.log('coming here')

   try {
      const uid = this.auth.currentUser?.uid;
      if (!uid) {
        console.error('User not logged in');
        return;
      }
      const uniqueId = this.generateUniqueId(8);

      // ✅ Add ID to project details
      const newProject = {
        id: uniqueId,
        uid: uid,
        companyInfo: this.projectForm.value,
      };
      const userRef = doc(this.firestore, 'user-data', uid);
      const userSnap = await getDoc(userRef);

      if (userSnap.exists()) {
        // ✅ Append new project to projects array
        await updateDoc(userRef, {
          projects: arrayUnion(newProject),
        });
        console.log('Project added to existing user document ✅');
      } else {
        // ⚠️ Should not happen (user doc is created on login)
        console.error('User document not found');
      }

      // Reset modal fields after save
      const modal = document.getElementById("projectDetailsModal") as HTMLDialogElement;
      if (modal) {
        modal.close();
      }
      this.projectForm.reset();
    } catch (error) {
      console.error('Error saving project details:', error);
    }
  }

  private generateUniqueId(length: number): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }
  goToAnalysis(projectId: string) {
  // replaceUrl true will replace current history entry with the analysis route
    this.router.navigate(['/dashboard', 'analysis', projectId], { replaceUrl: false });
    // notes: use replaceUrl:true if you want to avoid adding history entries
  }
  ngOnDestroy() {
    // ✅ Clean up listener when component is destroyed
    if (this.unsubscribe) {
      this.unsubscribe();
    }
  }
}
