import { Component } from '@angular/core';

@Component({
  selector: 'app-main-dash',
  standalone: false,
  templateUrl: './main-dash.component.html',
  styleUrl: './main-dash.component.css',
})
export class MainDashComponent {
  uploadedFiles: File[] = [];
  projects: any[] = [];
  projectDetails = this.getEmptyProject();
  getEmptyProject() {
    return {
      companyInfo: {
        name: '',
        industry: '',
        fundingStage: '',
        tagline: '',
      },
      performanceMetrics: {
        revenue: '',
        revenueGrowth: '',
        users: '',
        userGrowth: '',
      },
      scoring: {
        score: '',
      },
      additionalInfo: {
        lastUpdated: '',
      },
    };
  }

  ngOnInit() {
    const stored = localStorage.getItem('projects');
    if (stored) {
      this.projects = JSON.parse(stored);
    }
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

  saveDetails() {
    this.projects.push(JSON.parse(JSON.stringify(this.projectDetails)));
    localStorage.setItem('projects', JSON.stringify(this.projects));
    console.log('Saved Projects:', this.projects);
    this.projectDetails = this.getEmptyProject();
  }
}
