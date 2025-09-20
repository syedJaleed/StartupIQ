import { Component } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Auth, User, onAuthStateChanged } from '@angular/fire/auth';
import {
  Firestore,
  doc,
  getDoc,
  setDoc,
  updateDoc,
  arrayUnion,
  onSnapshot,
} from '@angular/fire/firestore';
import {
  Storage,
  ref,
  uploadBytes,
  getDownloadURL,
} from '@angular/fire/storage';

@Component({
  selector: 'app-project-analysis',
  standalone: false,
  templateUrl: './project-analysis.component.html',
  styleUrl: './project-analysis.component.css',
})
export class ProjectAnalysisComponent {
  uploadedFiles: File[] = [];
  savedFiles: string[] = [];
  projectId: string | null = null;
  project: any = null;
  dropBoxOpen = false;

  unsubscribe: (() => void) | null = null;
  syncJob: any = null; // 👈 holds realtime sync job data

  steps = [
    { key: 'file_analysis', label: 'File Analysis' },
    { key: 'team_agent', label: 'Team Agent' },
    { key: 'financial_stats_agent', label: 'Financial Stats Agent' },
    { key: 'final_agent', label: 'Final Agent' },
  ];

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private auth: Auth,
    private firestore: Firestore,
    private storage: Storage
  ) {}

  async ngOnInit() {
    onAuthStateChanged(this.auth, (user: User | null) => {
      if (user) {
        console.log('User is logged in:', user.uid);
        this.getProjectDetails(user.uid);
      } else {
        console.error('User not logged in');
      }
    });
  }

  async getProjectDetails(uid: string) {
    this.projectId = this.route.snapshot.paramMap.get('id');
    const userRef = doc(this.firestore, 'user-data', uid);
    const userSnap = await getDoc(userRef);

    if (userSnap.exists()) {
      const data = userSnap.data();
      const projects = data['projects'] || [];
      this.project = projects.find((p: any) => p.id === this.projectId);
      console.log('Loaded project:', this.project);
      if (this.projectId) {
        this.listenToFileUpdates(this.projectId);
        this.listenToSyncJob(this.projectId); // 👈 listen for sync updates
      }
    }
  }

  private listenToFileUpdates(projectId: string) {
    const fileDocRef = doc(this.firestore, 'files-map', projectId);
    this.unsubscribe = onSnapshot(fileDocRef, (docSnap) => {
      if (docSnap.exists()) {
        const data = docSnap.data();
        this.savedFiles = data['fileLinks'] || [];
        console.log('Realtime file updates:', this.savedFiles);
      } else {
        this.savedFiles = [];
      }
    });
  }

  private listenToSyncJob(projectId: string) {
    const syncDocRef = doc(this.firestore, 'sync-job', projectId);
    onSnapshot(syncDocRef, (docSnap) => {
      if (docSnap.exists()) {
        this.syncJob = docSnap.data();
        console.log('Realtime sync job:', this.syncJob);
      } else {
        this.syncJob = null;
      }
    });
  }

  openDropBox() {
    this.dropBoxOpen = !this.dropBoxOpen;
  }

  onFilesSelected(event: any) {
    const files: FileList = event.target.files;
    this.addFiles(files);
  }

  onFileDrop(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer?.files) {
      this.addFiles(event.dataTransfer.files);
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }

  private addFiles(files: FileList) {
    for (let i = 0; i < files.length; i++) {
      if (!this.uploadedFiles.find((f) => f.name === files[i].name)) {
        this.uploadedFiles.push(files[i]);
      }
    }
  }

  removeFile(index: number) {
    this.uploadedFiles.splice(index, 1);
  }

  getFileSize(size: number): string {
    if (size < 1024) return size + ' B';
    if (size < 1024 * 1024) return (size / 1024).toFixed(2) + ' KB';
    return (size / (1024 * 1024)).toFixed(2) + ' MB';
  }

  async saveFiles() {
    if (!this.projectId || this.uploadedFiles.length === 0) {
      console.error('No project selected or no files uploaded');
      return;
    }

    try {
      const uid = this.auth.currentUser?.uid;
      if (!uid) {
        console.error('User not logged in');
        return;
      }

      const fileLinks: string[] = [];

      for (const file of this.uploadedFiles) {
        const storagePath = `project-files/${this.projectId}/${Date.now()}-${
          file.name
        }`;
        const storageRef = ref(this.storage, storagePath);

        await uploadBytes(storageRef, file);
        const downloadURL = await getDownloadURL(storageRef);
        fileLinks.push(downloadURL);
      }

      const fileDocRef = doc(this.firestore, 'files-map', this.projectId);
      const docSnap = await getDoc(fileDocRef);

      if (docSnap.exists()) {
        await updateDoc(fileDocRef, {
          fileLinks: arrayUnion(...fileLinks),
          lastUpdated: new Date(),
        });
      } else {
        await setDoc(fileDocRef, {
          projectId: this.projectId,
          fileLinks,
          uploadedAt: new Date(),
        });
      }

      this.uploadedFiles = [];
    } catch (error) {
      console.error('Error saving files:', error);
    }
  }

  navDashboard() {
    this.router.navigate(['dashboard']);
  }

  getFileNameFromUrl(url: string): string {
    const withoutParams = url.split('?')[0];
    const parts = withoutParams.split('/');
    return decodeURIComponent(parts[parts.length - 1]);
  }

  async applySync() {
    if (!this.projectId) {
      console.error('No projectId found');
      return;
    }
    const syncDocRef = doc(this.firestore, 'sync-job', this.projectId);

    const syncData = {
      current_step: 'file_analysis',
      current_step_status: 'not_started',
      id: this.projectId,
      percentage: 0,
      status: 'not_started',
    };

    try {
      await setDoc(syncDocRef, syncData, { merge: true });
    } catch (error) {
      console.error('Error applying sync:', error);
    }
  }

  isStepActive(stepKey: string): boolean {
    if (!this.syncJob) return false;

    // Highlight current step
    if (this.syncJob.current_step === stepKey) {
      return true;
    }

    // If current step is completed, highlight past steps
    const currentIndex = this.steps.findIndex(
      (s) => s.key === this.syncJob.current_step
    );
    const stepIndex = this.steps.findIndex((s) => s.key === stepKey);

    if (
      this.syncJob.current_step_status === 'completed' &&
      stepIndex < currentIndex
    ) {
      return true;
    }

    return false;
  }

  isStepCompleted(stepKey: string): boolean {
    if (!this.syncJob) return false;

    const stepIndex = this.steps.findIndex((s) => s.key === stepKey);
    const currentIndex = this.steps.findIndex(
      (s) => s.key === this.syncJob.current_step
    );

    return (
      stepIndex < currentIndex ||
      (this.syncJob.current_step === stepKey &&
        this.syncJob.current_step_status === 'completed')
    );
  }

  ngOnDestroy() {
    if (this.unsubscribe) this.unsubscribe();
  }
}
