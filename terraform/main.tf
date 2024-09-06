terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "5.14.0"
    }
  }
}


provider "google" {
  credentials = file("C:/Users/ethan/git/Full_Chess_App/Chess_Model/terraform/secret.json")
  region      = var.region
  project = var.project_id
}


resource "google_service_account" "my_service_account" {
  account_id   = "chess-model-svc-acct"
  display_name = "Chess Model bucket actor"
}
resource "google_service_account_key" "my_service_account_key" {
  service_account_id = google_service_account.my_service_account.name
}
resource "kubernetes_secret" "my_gcp_secret" {
  metadata {
    name = "my-gcp-secret"
  }

  data = {
    "key.json" = google_service_account_key.my_service_account_key.private_key
  }
}

resource "google_project_iam_member" "service_account_storage_object_creator" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.my_service_account.email}"
}

resource "google_container_cluster" "gke_cluster" {
  name     = var.cluster_name
  location = var.region

  remove_default_node_pool = true
  initial_node_count = 1
  deletion_protection = false
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "model"
  location   = var.region
  cluster    = google_container_cluster.gke_cluster.name
  node_count = 1

  node_config {
    machine_type = "n1-highmem-4"

    # Specify the GPU and its count
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
    disk_size_gb = var.machine_size
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    preemptible = true

    # Ensure that the Google-provided container-optimized OS images with the necessary NVIDIA drivers are used
    image_type = "COS_CONTAINERD"
  }
}


provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "gke_full-chess_us-central1_chess-app"
}

resource "kubernetes_pod" "high_resource_ml_pod" {
  metadata {
    name = "self-training-pod"
    labels = {
      app = "ml-model"
    }
  }

  spec {
    container {
      image = "ethancruz/chess_model:v1.0.10"
      name  = "self-training"

      resources {
        requests = {
          memory = "24Gi"
          cpu    = "3000m"
          "nvidia.com/gpu" = 1  # Requesting 1 GPU
        }
        limits = {
          memory = "30Gi"
          cpu    = "4000m"
          "nvidia.com/gpu" = 1  # Limiting to 1 GPU
        }
      }

      volume_mount {
        mount_path = "/var/secrets/google"
        name       = "gcp-key"
        read_only  = true
      }
    }

    volume {
      name = "gcp-key"
      secret {
        secret_name = kubernetes_secret.my_gcp_secret.metadata[0].name
      }
    }

    # Optional: Add node affinity or tolerations here if needed
  }
}


resource "google_storage_bucket" "my_bucket" {
  name     = var.bucket_name
  location = var.bucket_location
  storage_class = var.storage_class
  force_destroy = true
}

