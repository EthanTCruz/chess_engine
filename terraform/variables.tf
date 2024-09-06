variable "region"{
    type = string
    default = "us-central1 (Iowa)"
}

variable "image"{
    type = string
    default = "us-central1 (Iowa)"
}

variable "zone"{
    type = string
    default = "us-central1-a"
}

variable "machine_type"{
    type = string
    default = "n1-standard-8 (8 vCPU, 4 core, 30 GB memory)"
}

variable "provisioning"{
    type = string
    default = "Spot"
}

variable "service_account" {
  type = string
}
variable "project_id"{
    type = string
}
variable "cluster_name" {
  type = string
}
variable "bucket_name" {
  type = string
}
variable "bucket_location" {
  type = string
}
variable "storage_class" {
  type = string
}
variable "machine_size" {
  type = number
}