// ContactForm component for form-toy.
//
// Renders a contact form with name, email, and message fields.
// Delegates validation to useFormValidation and shows inline error
// messages beside each field. Submits to /api/contact on success.

import React, { useState } from "react";
import { useFormValidation } from "./useFormValidation.js";

interface ContactFormData {
  name: string;
  email: string;
  message: string;
}

const INITIAL_VALUES: ContactFormData = { name: "", email: "", message: "" };

export function ContactForm(): React.ReactElement {
  const [values, setValues] = useState<ContactFormData>(INITIAL_VALUES);
  const [submitted, setSubmitted] = useState(false);

  const { errors, validate, clearError } = useFormValidation(values);

  function handleChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ): void {
    const { name, value } = e.target;
    setValues((prev) => ({ ...prev, [name]: value }));
    clearError(name as keyof ContactFormData);
  }

  async function handleSubmit(e: React.FormEvent): Promise<void> {
    e.preventDefault();
    if (!validate()) return;
    await fetch("/api/contact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    setSubmitted(true);
  }

  if (submitted) return <p>Thanks, we will be in touch!</p>;

  return (
    <form onSubmit={handleSubmit} noValidate>
      <label>
        Name
        <input name="name" value={values.name} onChange={handleChange} />
        {errors.name && <span className="error">{errors.name}</span>}
      </label>
      <label>
        Email
        <input name="email" type="email" value={values.email} onChange={handleChange} />
        {errors.email && <span className="error">{errors.email}</span>}
      </label>
      <label>
        Message
        <textarea name="message" value={values.message} onChange={handleChange} />
        {errors.message && <span className="error">{errors.message}</span>}
      </label>
      <button type="submit">Send</button>
    </form>
  );
}
