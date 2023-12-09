extern void jacubi();
extern void jacubi_redblack();
extern void sor();
extern void sor_separated();
extern void conjugate_gradient();


int main() {

	switch (4)
	{
	case 0:
		jacubi();
		break;
	case 1:
		jacubi_redblack();
		break;
	case 2:
		sor();
		break;
	case 3:
		sor_separated();
		break;
	case 4:
		conjugate_gradient();
		break;
	default:
		break;
	}

	return 0;
}
