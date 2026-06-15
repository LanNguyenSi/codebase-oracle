// useFormValidation hook for form-toy.
//
// Validates a ContactFormData object against field rules and returns
// an errors map plus a validate() trigger and a clearError() helper.
// The hook is intentionally rule-based (no schema library) so the
// fixture stays dependency-free.

import { useState } from "react";

interface ContactFormData {
  name: string;
  email: string;
  message: string;
}

type FormErrors = Partial<Record<keyof ContactFormData, string>>;

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

interface UseFormValidationReturn {
  errors: FormErrors;
  validate: () => boolean;
  clearError: (field: keyof ContactFormData) => void;
}

export function useFormValidation(values: ContactFormData): UseFormValidationReturn {
  const [errors, setErrors] = useState<FormErrors>({});

  function validate(): boolean {
    const next: FormErrors = {};

    if (!values.name.trim()) {
      next.name = "Name is required.";
    } else if (values.name.trim().length < 2) {
      next.name = "Name must be at least 2 characters.";
    }

    if (!values.email.trim()) {
      next.email = "Email is required.";
    } else if (!EMAIL_REGEX.test(values.email)) {
      next.email = "Please enter a valid email address.";
    }

    if (!values.message.trim()) {
      next.message = "Message is required.";
    } else if (values.message.trim().length < 10) {
      next.message = "Message must be at least 10 characters.";
    }

    setErrors(next);
    return Object.keys(next).length === 0;
  }

  function clearError(field: keyof ContactFormData): void {
    setErrors((prev) => {
      const next = { ...prev };
      delete next[field];
      return next;
    });
  }

  return { errors, validate, clearError };
}
